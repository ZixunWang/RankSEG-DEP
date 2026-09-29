# Prepare Datasets

```
data_dir
|—— cityscapes
|   |—— leftImg8bit
|   |   |—— train
|   |   |—— val
|   |—— gtFine
|   |   |—— train
|   |   |—— val
|—— ade
|   |—— ADEChallengeData2016
|   |   |—— images
|   |   |   |—— training
|   |   |   |—— validation
|   |   |—— annotations
|   |   |   |—— training
|   |   |   |—— validation
|—— land
|   |—— train
|—— lits
|   |—— train
|—— kits
|   |—— train
```

## Cityscapes
* Step 1: Download the dataset from [here](https://www.cityscapes-dataset.com)
* Step 2: Run the following from `MMSegmentation`

  ```
  python tools/dataset_converters/cityscapes.py data/cityscapes
  ```

## ADE20K
* Download the dataset from [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)


## DeepGlobe Land
* Run the following
  ```
  python datas/prepare/prepare_deepglobe_land.py path/to/data_dir
  ```

## LiTS
* Run the following
  ```
  python datas/prepare/prepare_lits_kits.py path/to/data_dir lits
  ```

## KiTS
* Run the following
  ```
  python datas/prepare/prepare_lits_kits.py path/to/data_dir kits
  ```
