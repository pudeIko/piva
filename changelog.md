# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic 
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed


## [2.2.0]

### Changed

- notation used in the linking functionality

### Fixed

- dependency issues due to reliance on outdated library versions

- appearance issues on Windows


## [2.1.0]

### Added

- `CustomWidget`: a framework for importing custom, user-defined widgets, along 
with templates and documentation on implementation
- introduces a framework for implementing custom data loaders
- `data_viewer_4D` class for visualization of *x-y* raster scans
- higher-level analysis methods
- showcase and more extensive documentation

### Changed

### Deprecated

### Removed

### Fixed

- bugs in experimental logbooks and documentation
- bugs in normalization and conversion methods 
- missing jupyter-lab templates in built version


## [2.0.1] = 2023-11-12

### Fixed

- Example data feature directing to correct datafile


## [2.0.0] = 2023-11-04

### Added

- `DataLoader` classes for MERLIN (ALS) and URANOS (Solaris) beamlines
- experimental logbook feature for implemented `DataLoaders`
- data provenance functionality, to track modifications in original data sets
- extended options for data normalization
- linking viewers functionality, allowing for simultaneous inspection of 
similar datasets
- automated tests
- full in-code and finished docstring documentation


### Changed

- JuPyter functionalities from JuPyter-notebook to JuPyterLab environment
- generalizes superclass `Dataloader` and simplifies Dataloaders from different 
beamlines
- changes `Dataset` to directly inherit from Namespace
- unifies fitters into `Fitter` class, from which `MDCFitter` and `EDCFitter` 
inherit
 

### Removed
- `DataLoader` classes for MAESTRO (ALS)

### Fixed

- minor bugs in `set_metadata_window`
- bugs in `MDCFitter` and `EDCFitter`


## [1.1.0] = 2022-11-28

### Added

- Horizontal scrollbar now appears when needed in data_browser.

### Changed

- root directory of a tree view window can be changed with `Ctrl+O` instead of 
fixed `home_dir` 


## [1.0.4] = 2022-11-26

### Added

- Proper k-space conversion routine for both manipulator & photon energy scans. 
Uses a "rescaling" approach to account for momentum conservation, instead of 
previous simple swapping of axes values.

### Changed

- Python version >3.8 is now required, rather than previous >3.7.

### Fixed

- Issue #28: colorscales now work as expected again.
- Appearance issues of `QTabWidget` on Windows.
- Several small bugfixes.

## [1.0.3] = 2022-11-21

### Fixed

- Issue #27 & #33: Save button now working properly.
- Issue #29: Commented a part of the code in `cmaps.py` which was attempting 
  to load `data-slicer` colormaps. See issue #29. PIT now opening from piva.
- Issue #30: Inverting colorscale in 2D viewer.
- Issue #31: c-axis field accepts values beyond 10 Angstrom.
- Issue #32: metadata-window now has a scrollbar.

## [1.0.2] = 2022-11-17

### Added

- "Open in PIT" button

## [1.0.1] = 2022-11-17

### Changed

- `db` function (also `python -m piva.main`) now displays current version 
  number on startup.

### Fixed

- wheel and source packages differed on pypi.

## [1.0.0] = 2022-11-17

First official release.
