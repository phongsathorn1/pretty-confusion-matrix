# Pretty Confusion Matrix for Python
The Pretty Confusion Matrix in Python with MATLAB like style, using seaborn and matplotlib.

This repository was forked and modified from [Wagner's Pretty print confusion matrix](https://github.com/wcipriano/pretty-print-confusion-matrix).

---
**Example**:

<img src="Screenshots/conf_matrix_default.png" width="650" alt="Example of Pretty Confusion Matrix">

## Installation
- Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Plot from numpy `x` and `y` vectors**
   ```python
   from pretty_cm import plot_from_data

   y_test = np.array([1,2,3,4,5, 1,2,3,4,5, ...])
   predic = np.array([1,2,4,3,5, 1,2,3,4,4, ...])

   plot_from_data(y_test, predic)
   ```

- **Plot from numpy confusion matrix**
   ```python
   from pretty_cm import plot_from_confusion_matrix

   cm = np.array([[13,  0,  1,  0,  2,  0],
                  [ 0, 50,  2,  0, 10,  0],
                  [ 0, 13, 16,  0,  0,  3],
                  [ 0,  0,  0, 13,  1,  0],
                  [ 0, 40,  0,  1, 15,  0],
                  [ 0,  0,  0,  0,  0, 20]])

   plot_from_confusion_matrix(cm)
   ```

- **Plot with custom labels**
   ```python
   plot_from_data(y_test, predic, 
                  columns=["Dog", "Cat", "Potato", "Car", "IU <3"])
   ```

   Result:

   <img src="Screenshots/conf_matrix_custom_labels.png" width="450" alt="Example of Pretty Confusion Matrix">

## Licensing
The Pretty Confusion Matrix is licensed under Apache License, Version 2.0. see [License](LICENSE) for full license text.

## References:
1. MATLAB confusion matrix

   - https://www.mathworks.com/help/nnet/ref/plotconfusion.html
   
   - https://www.mathworks.com/help/examples/nnet/win64/PlotConfusionMatrixUsingCategoricalLabelsExample_02.png
