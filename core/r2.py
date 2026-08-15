"""Cloudflare R2 (S3-compatible) mirroring for activation files.

Why this exists
---------------
A full-layer run over 10 emotions with Qwen2.5-32B is ~9 GiB of pooled
activations (see ``--dry-run`` for the exact figure). That is too much to keep
shuttling to a laptop, so extraction can mirror each chunk to R2 as soon as it is
written, optionally deleting the local copy afterwards.

Credentials come from the environment (see ``.env.example``)::

    R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET

plus an endpoint, from any one of::

    R2_ENDPOINT        <- the name the shared scripts/ workflow's r2.env exports
    R2_ENDPOINT_URL    <- alias
    R2_ACCOUNT_ID      <- endpoint derived as https://<id>.r2.cloudflarestorage.com

Accepting ``R2_ENDPOINT`` matters: ``run-emotionvectors`` forwards
``scripts/r2.env`` to the pod and sources it, and that file exports
``R2_ENDPOINT``. If we only looked for ``R2_ENDPOINT_URL``, ``r2_sync="auto"``
would quietly decide R2 was unconfigured and keep 8 GiB of activations on an
ephemeral pod.

CLI::

    python -m core.r2 push outputs/<run>/activations --prefix runs/<run>/activations
    python -m core.r2 pull outputs/<run>/activations --prefix runs/<run>/activations
    python -m core.r2 ls --prefix runs/<run>
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REQUIRED_ENV = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")

#: Endpoint variable names we accept, in precedence order. ``R2_ENDPOINT`` is what
#: the shared ``scripts/`` workflow's ``r2.env`` exports.
ENDPOINT_ENV = ("R2_ENDPOINT", "R2_ENDPOINT_URL")


class R2ConfigError(RuntimeError):
    pass


def resolve_endpoint() -> str | None:
    """The R2 endpoint URL from the environment, or ``None`` if unset."""
    for name in ENDPOINT_ENV:
        value = os.environ.get(name)
        if value:
            return value.rstrip("/")
    account = os.environ.get("R2_ACCOUNT_ID")
    if account:
        return f"https://{account}.r2.cloudflarestorage.com"
    return None


def r2_available() -> tuple[bool, str]:
    """Whether R2 mirroring can run, plus a reason if not."""
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        return False, f"missing environment variables: {', '.join(missing)}"
    if not resolve_endpoint():
        return False, f"set one of {', '.join(ENDPOINT_ENV)} or R2_ACCOUNT_ID"
    try:
        import boto3  # noqa: F401
    except ImportError:
        return False, "boto3 is not installed (pip install boto3)"
    return True, "ok"


@dataclass
class R2Client:
    """Thin wrapper over the handful of S3 operations we need."""

    bucket: str
    endpoint_url: str
    client: object

    @classmethod
    def from_env(cls) -> "R2Client":
        ok, reason = r2_available()
        if not ok:
            raise R2ConfigError(f"R2 is not configured: {reason}")
        import boto3
        from botocore.config import Config

        endpoint = resolve_endpoint()
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            # R2 ignores the region but boto3 requires one; 'auto' is Cloudflare's.
            region_name=os.environ.get("R2_REGION", "auto"),
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"},
                # R2 does not support the newer default checksum algorithms.
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )
        return cls(bucket=os.environ["R2_BUCKET"], endpoint_url=endpoint, client=client)

    # -- listing ----------------------------------------------------------- #

    def list_objects(self, prefix: str) -> dict[str, int]:
        """``{key: size}`` for everything under ``prefix``."""
        out: dict[str, int] = {}
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                out[obj["Key"]] = int(obj["Size"])
            if not resp.get("IsTruncated"):
                return out
            token = resp.get("NextContinuationToken")

    # -- transfer ---------------------------------------------------------- #

    def upload_file(self, local: str | Path, key: str) -> None:
        self.client.upload_file(str(local), self.bucket, key)

    def head_size(self, key: str) -> int | None:
        """Byte size of one object, or ``None`` if it does not exist.

        Used to verify an upload landed before deleting the local copy.
        """
        try:
            return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])
        except Exception:
            return None

    def download_file(self, key: str, local: str | Path) -> None:
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local))

    # -- sharing ----------------------------------------------------------- #

    def presign(self, key: str, expires_seconds: int = 604800) -> str:
        """A time-limited HTTPS URL granting read access to one object.

        Lets a collaborator download without a Cloudflare account or any
        credentials. The URL embeds a signature derived from *your* access key, so
        treat it as a secret: anyone holding it has read access to that object
        until it expires.

        R2 caps presigned-URL lifetime at 7 days (604800s), which is also the
        default here.
        """
        if not 1 <= expires_seconds <= 604800:
            raise ValueError(
                f"expires_seconds must be 1..604800 (R2's 7-day maximum), got {expires_seconds}"
            )
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )

    def presign_prefix(
        self,
        prefix: str,
        expires_seconds: int = 604800,
    ) -> list[tuple[str, int, str]]:
        """``[(key, size, url), ...]`` for every object under ``prefix``.

        A whole run is hundreds of chunk files, so this is mainly useful for
        sharing the small stuff -- a ``results/`` tree, one layer, a manifest. For
        a full 8 GiB activation set, hand out a read-only API token instead
        (Option 1 in the README) rather than hundreds of links.
        """
        return [
            (key, size, self.presign(key, expires_seconds))
            for key, size in sorted(self.list_objects(prefix).items())
        ]

    def sync_up(
        self,
        local_dir: str | Path,
        prefix: str,
        delete_local: bool = False,
        verbose: bool = True,
    ) -> dict[str, int]:
        """Upload everything under ``local_dir`` that is not already in R2.

        "Already there" means same key and same byte size. Activation chunks are
        immutable once written, so size is a sufficient check and avoids hashing
        gigabytes on every resume.
        """
        local_dir = Path(local_dir)
        remote = self.list_objects(prefix.rstrip("/") + "/")
        uploaded = skipped = 0
        n_bytes = 0
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            key = f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}"
            size = path.stat().st_size
            if remote.get(key) == size:
                skipped += 1
            else:
                self.upload_file(path, key)
                uploaded += 1
                n_bytes += size
                if verbose:
                    print(f"  uploaded {key} ({size / 1024**2:.1f} MiB)")
            if delete_local:
                path.unlink()
        return {"uploaded": uploaded, "skipped": skipped, "bytes": n_bytes}

    def sync_down(self, local_dir: str | Path, prefix: str, verbose: bool = True) -> dict[str, int]:
        """Download everything under ``prefix`` that is missing locally."""
        local_dir = Path(local_dir)
        remote = self.list_objects(prefix.rstrip("/") + "/")
        downloaded = skipped = 0
        n_bytes = 0
        for key, size in sorted(remote.items()):
            rel = key[len(prefix.rstrip("/")) + 1 :]
            target = local_dir / rel
            if target.exists() and target.stat().st_size == size:
                skipped += 1
                continue
            self.download_file(key, target)
            downloaded += 1
            n_bytes += size
            if verbose:
                print(f"  downloaded {key} ({size / 1024**2:.1f} MiB)")
        return {"downloaded": downloaded, "skipped": skipped, "bytes": n_bytes}


def make_chunk_uploader(
    prefix: str,
    activations_dir: str | Path,
    delete_local: bool = False,
    verbose: bool = False,
):
    """Build an ``on_chunk_written`` callback for :class:`~core.activation_store.ActivationWriter`.

    Failures are reported but never abort extraction: GPU time is the scarce
    resource, and anything missed is picked up by a later ``python -m core.r2 push``.
    """
    client = R2Client.from_env()
    activations_dir = Path(activations_dir)

    def upload(paths: list[Path]) -> None:
        uploaded: list[Path] = []
        for path in paths:
            key = f"{prefix.rstrip('/')}/{Path(path).relative_to(activations_dir).as_posix()}"
            local_size = Path(path).stat().st_size
            try:
                client.upload_file(path, key)
                if delete_local:
                    # Deleting the only local copy: confirm the object is actually
                    # there at the right size before doing it. An interrupted or
                    # truncated PUT that we did not verify would otherwise lose the
                    # chunk outright -- and a chunk costs real GPU time to recreate.
                    remote_size = client.head_size(key)
                    if remote_size != local_size:
                        raise IOError(
                            f"size mismatch after upload: local {local_size} != "
                            f"remote {remote_size if remote_size is not None else 'missing'}"
                        )
                uploaded.append(Path(path))
                if verbose:
                    print(f"  [r2] {key}")
            except Exception as exc:
                # Keep every local file for this chunk: the final sweep or a later
                # `r2 push` retries it. Never delete on a partial failure.
                print(f"  [r2] WARNING failed to upload {key}: {exc}")
                print("  [r2] keeping local copies for this chunk; will retry in the final sweep")
                return

        if delete_local:
            # Only delete the tensor file. The index parquet is tiny and is what
            # lets a resumed run know these examples are done without touching R2.
            for path in uploaded:
                if path.suffix == ".safetensors":
                    path.unlink(missing_ok=True)

    return upload


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Mirror activations to/from Cloudflare R2")
    parser.add_argument("action", choices=["push", "pull", "ls", "check", "share"])
    parser.add_argument("local_dir", nargs="?", help="local directory (push/pull)")
    parser.add_argument("--prefix", required=False, default="", help="key prefix in the bucket")
    parser.add_argument("--delete-local", action="store_true", help="remove local files after upload")
    parser.add_argument(
        "--expires-hours",
        type=float,
        default=168.0,
        help="presigned-URL lifetime for 'share' (max 168 = R2's 7-day cap)",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=50,
        help="refuse to mint more than this many URLs at once (guards against "
             "accidentally presigning a whole 8 GiB run)",
    )
    args = parser.parse_args()

    if args.action == "check":
        ok, reason = r2_available()
        print(f"R2 configured: {ok} ({reason})")
        if ok:
            client = R2Client.from_env()
            print(f"  bucket   : {client.bucket}")
            print(f"  endpoint : {client.endpoint_url}")
        return

    client = R2Client.from_env()
    if args.action == "ls":
        objects = client.list_objects(args.prefix)
        total = sum(objects.values())
        for key, size in sorted(objects.items()):
            print(f"{size:>14,}  {key}")
        print(f"{len(objects)} objects, {total / 1024**3:.2f} GiB")
        return

    if args.action == "share":
        if not args.prefix:
            parser.error("share requires --prefix")
        expires = int(args.expires_hours * 3600)
        objects = client.list_objects(args.prefix)
        if not objects:
            print(f"no objects under {args.prefix!r} — nothing to share")
            return
        if len(objects) > args.max_urls:
            print(
                f"{len(objects)} objects under {args.prefix!r} exceeds --max-urls="
                f"{args.max_urls}.\n"
                "Presigned URLs are per-object, so this is the wrong tool for a whole "
                "activation set.\nHand out a read-only R2 API token instead (see README "
                '"Sharing activations"), or\nnarrow --prefix, or raise --max-urls '
                "deliberately."
            )
            raise SystemExit(1)

        print(
            f"# {len(objects)} presigned URL(s), valid {args.expires_hours:g}h "
            f"({expires}s) from now.\n"
            "# Treat as secrets: each grants read access to that object until it expires.\n"
        )
        for key, size, url in client.presign_prefix(args.prefix, expires):
            print(f"# {key}  ({size / 1024**2:.1f} MiB)")
            print(url)
            print()
        return

    if not args.local_dir:
        parser.error(f"{args.action} requires a local_dir")
    if not args.prefix:
        parser.error(f"{args.action} requires --prefix")

    if args.action == "push":
        stats = client.sync_up(args.local_dir, args.prefix, delete_local=args.delete_local)
        print(f"uploaded {stats['uploaded']}, skipped {stats['skipped']}, "
              f"{stats['bytes'] / 1024**3:.2f} GiB transferred")
    else:
        stats = client.sync_down(args.local_dir, args.prefix)
        print(f"downloaded {stats['downloaded']}, skipped {stats['skipped']}, "
              f"{stats['bytes'] / 1024**3:.2f} GiB transferred")


if __name__ == "__main__":
    _main()
