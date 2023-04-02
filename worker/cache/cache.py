import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from nataili import disable_progress
from worker.logger import logger

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources


def get_package():
    pkg = importlib_resources.files("annotator")
    return pkg


def get_cache_directory():
    """The nataili specific directory for caching."""
    AIWORKER_CACHE_HOME = os.environ.get("AIWORKER_CACHE_HOME")
    base_dir = ""
    if AIWORKER_CACHE_HOME:
        base_dir = AIWORKER_CACHE_HOME
    else:
        base_dir = os.environ.get("XDG_CACHE_HOME", os.path.join(Path.home(), ".cache/"))
    return os.path.join(base_dir, "nataili")


class Cache:
    def __init__(self, cache_name, cache_subname=None, cache_parentname=None):
        """
        :param cache_name: Name of the cache
        :param cache_subname: Subfolder in the cache
        :param cache_parentname: Parent folder of the cache
        Examples:
        cache = Cache("test", "sub", "parent")
        path = self.path + "/parent/test/sub"

        cache = Cache("test", "sub")
        path = self.path + "/test/sub"

        cache = Cache("test")
        path = self.path + "/test"

        cache = Cache("test", cache_parentname="parent")
        path = self.path + "/parent/test"

        If cache file does not exist it is created
        If cache folder does not exist it is created
        """
        self.path = get_cache_directory()
        if cache_parentname:
            self.path = os.path.join(self.path, cache_parentname)
        self.cache_dir = os.path.join(self.path, cache_name)
        if cache_subname:
            self.cache_dir = os.path.join(self.cache_dir, cache_subname)
        self.cache_db = os.path.join(self.cache_dir, "cache.db")
        logger.debug(f"Cache file: {self.cache_db}")
        logger.debug(f"Cache dir: {self.cache_dir}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.cache_db)
        self.cursor = self.conn.cursor()
        self.create_sqlite_db()

    def list_dir(self, input_directory, extensions=[".webp"]):
        """
        List all files in a directory
        :param input_directory: Directory to list
        :param extensions: List of extensions to filter for
        :return: List of files
        """
        files = []
        for file in tqdm(os.listdir(input_directory), disable=disable_progress.active):
            if os.path.splitext(file)[1] in extensions:
                files.append(os.path.splitext(file)[0])
        return files

    def get_all(self):
        """
        Get all entries from the cache
        :return: List of all entries
        """
        self.cursor.execute("SELECT file FROM cache")
        return [x[0] for x in self.cursor.fetchall()]

    def get_all_export(self):
        self.cursor.execute("SELECT file, pil_hash FROM cache")
        return {x[0]: x[1] for x in self.cursor.fetchall()}

    def filter_list(self, input_list):
        """
        Filter a list
        :param input_list: List to filter
        :param filter_list: List to filter with
        :return: Filtered list
        """
        db_list = self.get_all()
        logger.info(f"Filtering {len(input_list)} files")
        logger.info(f"Filtering {len(db_list)} files")
        logger.info(f"Filtering {len(set(input_list) - set(db_list))} files")
        logger.info(f"First item in input_list: {input_list[0]}")
        logger.info(f"First item in db_list: {db_list[0]}")
        return list(set(input_list) - set(db_list))

    def hash_file(self, file_path):
        """
        Hash a file
        :param file_path: Path to the file
        :return: Hash of the file
        """
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def hash_pil_image(self, pil_image: Image.Image):
        """
        Hash a PIL image
        :param pil_image: PIL image
        :return: Hash of the PIL image
        """
        return hashlib.sha256(pil_image.tobytes()).hexdigest()

    def hash_pil_image_file(self, file_path):
        """
        Hash a PIL image
        :param file_path: Path to the file
        :return: Hash of the PIL image
        """
        pil_image = Image.open(file_path)
        return self.hash_pil_image(pil_image)

    def hash_files(self, files_list, input_directory, extensions=[".webp"]):
        """
        Hash all files in a directory
        :param input_directory: Directory to hash
        :return: List of hashes
        """
        pil_hashes = []
        file_hashes = []
        for file in tqdm(files_list, disable=disable_progress.active):
            for extension in extensions:
                file = file + extension
                file_path = os.path.join(input_directory, file)
                file_hash = self.hash_file(file_path)
                pil_image_hash = self.hash_pil_image_file(file_path)
                pil_hashes.append(pil_image_hash)
                file_hashes.append(file_hash)
        return pil_hashes, file_hashes

    def create_sqlite_db(self):
        """
        Create a sqlite database from the cache
        """
        self.cursor.execute("CREATE TABLE IF NOT EXISTS cache (file text, hash text, pil_hash text)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS file_index ON cache (file)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS hash_index ON cache (hash)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS pil_hash_index ON cache (pil_hash)")
        self.conn.commit()

    def add_sqlite_row(self, file: str, hash: str, pil_hash: str, commit=True):
        """
        Add a row to the sqlite database
        """
        self.cursor.execute("INSERT INTO cache VALUES (?, ?, ?)", (file, hash, pil_hash))
        if commit:
            self.conn.commit()

    def populate_sqlite_db(self, list_of_files: list):
        """
        Populate the sqlite database from the cache
        """
        # Populate sqlite database
        for file in list_of_files:
            self.add_sqlite_row(file["file"], file["hash"], file["pil_hash"], commit=False)
        self.conn.commit()

    def key_exists(self, key):
        """
        Check if a key exists in the cache
        """
        query = "SELECT hash, pil_hash FROM cache WHERE file=?"
        self.cursor.execute(query, (key,))
        if self.cursor.fetchone():
            return True
        return False

    def get(self, file: str = None, file_hash: str = None, pil_hash: str = None, no_return=False):
        """
        Get a file from the cache
        """
        if not any([file, file_hash, pil_hash]):
            raise ValueError("At least one value must be provided to search the database")
        file = os.path.splitext(file)[0] if file else None
        query = "SELECT hash, pil_hash FROM cache WHERE "
        conditions = []
        values = []
        if file:
            conditions.append("file=?")
            values.append(file)
        if file_hash:
            conditions.append("hash=?")
            values.append(file_hash)
        if pil_hash:
            conditions.append("pil_hash=?")
            values.append(pil_hash)

        query += " OR ".join(conditions)

        self.cursor.execute(query, tuple(values))
        result = self.cursor.fetchone()
        if result:
            if no_return:
                return True
            file_hash, pil_hash = result
            if file_hash:
                file_path = os.path.join(self.cache_dir, file_hash + ".npy")
                if os.path.exists(file_path):
                    return file_path
            if pil_hash:
                file_path = os.path.join(self.cache_dir, pil_hash + ".npy")
                if os.path.exists(file_path):
                    return file_path
        return None
