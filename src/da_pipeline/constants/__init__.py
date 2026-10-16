"""Names the pipeline agrees on, grouped by what they describe.

Four modules, because these are four different kinds of constant with four
different reasons to change:

``columns``
    Names of columns in the dataframes this pipeline builds. Ours to choose;
    changing one is a refactor.
``schema``
    The extract-file format: this project's own specification. Changing one is
    a change to the format, so it needs a schema version bump and a producer
    that writes it.
``outputs``
    Directory and file names a run writes. Changing one moves a user's results.
``display``
    Words and proportions that appear in figures. Changing one alters what a
    reader sees, not what the code does.

Import the module, not the symbols, so that every use site says which kind of
name it is reaching for::

    from da_pipeline.constants import columns as col
    from da_pipeline.constants import schema as sch

    df.group_by(col.CONDITION)                     # a column in our frame
    extract.dataset(sch.GROUP_OBSERVATIONS, ...)   # a name in the file format
"""

from . import columns, display, outputs, schema

__all__ = ['columns', 'display', 'outputs', 'schema']
