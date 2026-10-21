__all__ = [
    "FeatureProcessor",
    "RowParser",
    "collect_hdfs_file_paths",
    "hdfs_ls",
]


def __getattr__(name: str):
    if name == "FeatureProcessor":
        from .feature_processor import FeatureProcessor

        return FeatureProcessor
    if name in {"collect_hdfs_file_paths", "hdfs_ls"}:
        from .hdfs_source import collect_hdfs_file_paths, hdfs_ls

        return {"collect_hdfs_file_paths": collect_hdfs_file_paths, "hdfs_ls": hdfs_ls}[
            name
        ]
    if name == "RowParser":
        from .row_parser import RowParser

        return RowParser
    raise AttributeError(name)
