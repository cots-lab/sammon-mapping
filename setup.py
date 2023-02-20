import setuptools
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="sammon-mapping",
    version="0.0.2",
    author="Dilan Perera",
    packages= ['sammon'],
    long_description = long_description,
    long_description_content_type="text/markdown"
)