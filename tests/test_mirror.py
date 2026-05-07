import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock, patch

from apt_mirror.apt_mirror import RepositoryMirror
from apt_mirror.config import Config
from apt_mirror.download import DownloadFile
from apt_mirror.download.downloader import Downloader, DownloaderSettings
from apt_mirror.download.response import DownloadResponse
from apt_mirror.repository import BaseRepository, InvalidSignatureError


class TestRepositoryMirror(IsolatedAsyncioTestCase):
    REPOSITORY_URL = "http://test.example/repo"
    METADATA_PATH = Path("dists/test/main/binary-amd64/Packages.xz")
    POOL_PATH = Path("pool/main/p/pkg/pkg_1_amd64.deb")

    class _MockedDownloader(Downloader):
        RETRY_TIMEOUT = 0

        def __init__(
            self,
            *,
            settings: DownloaderSettings,
        ):
            self._release_requests = 0
            super().__init__(settings=settings)

        @staticmethod
        async def _stream_content(content: bytes) -> AsyncIterator[bytes]:
            yield content

        @asynccontextmanager
        async def stream(self, source_path: Path):
            if source_path.name == "InRelease":
                self._release_requests += 1

                if self._release_requests == 1:
                    content = b"first release\n"
                    yield DownloadResponse(
                        _stream=lambda: self._stream_content(content),
                        size=len(content),
                    )
                elif self._release_requests == 2:
                    content = b"release downloaded during recheck\n"
                    yield DownloadResponse(
                        _stream=lambda: self._stream_content(content),
                        size=len(content),
                    )
                else:
                    yield DownloadResponse(_stream=None, missing=True)
            elif source_path.name in {"Release", "Release.gpg"}:
                yield DownloadResponse(_stream=None, missing=True)
            elif source_path == TestRepositoryMirror.METADATA_PATH:
                content = b"metadata downloaded during recheck\n"
                yield DownloadResponse(
                    _stream=lambda: self._stream_content(content),
                    size=len(content),
                )
            elif source_path == TestRepositoryMirror.POOL_PATH:
                yield DownloadResponse(_stream=None, error="pool download failed")
            else:
                raise RuntimeError(f"Unexpected download path: {source_path}")

    @staticmethod
    def _snapshot_tree(root: Path) -> dict[Path, bytes]:
        return {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }

    @staticmethod
    def _get_config(root: Path, extra: dict[str, str] | None = None) -> Config:
        if not extra:
            extra = {}

        with NamedTemporaryFile("wt", encoding="utf-8") as fp:
            fp.write(
                "\n".join(
                    [
                        "set defaultarch amd64",
                        "set check_hashes off",
                        f"deb {TestRepositoryMirror.REPOSITORY_URL} test main",
                        f"mirror_path {TestRepositoryMirror.REPOSITORY_URL} repo",
                    ]
                    + [f"set {k} {v}" for k, v in extra.items()]
                )
                + "\n"
            )
            fp.flush()

            return Config(Path(fp.name), default_base_path=str(root))

    async def _create_mirror(
        self, repository: BaseRepository, config: Config
    ) -> RepositoryMirror:
        with patch(
            "apt_mirror.apt_mirror.DownloaderFactory.for_settings",
            side_effect=lambda *, settings: self._MockedDownloader(settings=settings),
        ):
            return await RepositoryMirror.create(
                repository,
                config,
                asyncio.Semaphore(1),
                asyncio.Semaphore(1),
                None,
                Mock(),
            )

    async def test_failed_retry_dists_unchanged(self):
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)

            config = self._get_config(root, {"release_files_retries": "1"})
            repository = config.repositories[self.REPOSITORY_URL]

            dists_path = (
                config.mirror_path
                / repository.get_mirror_path(config.encode_tilde)
                / "dists"
            )
            inrelease_path = dists_path / "test" / "InRelease"

            packages_path = (
                dists_path / "test" / "main" / "binary-amd64" / "Packages.xz"
            )
            packages_path.parent.mkdir(parents=True)

            inrelease_path.write_bytes(b"published release\n")
            packages_path.write_bytes(b"published packages\n")

            before = self._snapshot_tree(dists_path)
            with (
                patch.object(
                    repository,
                    "get_metadata_files",
                    return_value={DownloadFile.from_path(self.METADATA_PATH)},
                ),
                patch.object(
                    repository,
                    "get_pool_files",
                    return_value={DownloadFile.from_path(self.POOL_PATH)},
                ),
            ):
                mirror = await self._create_mirror(repository, config)
                result = await mirror.mirror()

            self.assertFalse(result)
            self.assertEqual(before, self._snapshot_tree(dists_path))

    async def test_release_gpg_retry(self):
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)

            config = self._get_config(root)
            repository = config.repositories[self.REPOSITORY_URL]
            mirror = await self._create_mirror(repository, config)
            inrelease_path = (
                config.skel_path
                / repository.get_mirror_path(config.encode_tilde)
                / "dists/test/InRelease"
            )

            with (
                patch.object(
                    RepositoryMirror,
                    "RELEASE_FILES_RETRY_TIMEOUT",
                    0,
                ),
                patch.object(
                    repository,
                    "validate_release_files",
                    side_effect=[
                        InvalidSignatureError(
                            "Unable to verify release file signature"
                        ),
                        None,
                    ],
                ) as validate_release_files,
            ):
                release_files = await mirror.download_release_files()

            self.assertTrue(release_files)
            self.assertEqual(validate_release_files.call_count, 2)
            self.assertEqual(
                inrelease_path.read_bytes(),
                b"release downloaded during recheck\n",
            )
