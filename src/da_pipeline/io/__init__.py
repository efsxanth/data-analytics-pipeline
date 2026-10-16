"""Everything that touches the filesystem.

Reading extract files, reading the run's JSON config, deciding where results
go, and writing CSVs. Isolating them here means the analysis modules take and
return dataframes only, which is what makes them testable without a disk.
"""
