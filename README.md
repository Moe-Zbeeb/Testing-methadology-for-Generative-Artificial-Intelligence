

## One-Class SVM in Python and C

### Description
This project showcases the implementation of a One-Class Support Vector Machine (SVM) for anomaly detection in both Python and C. The Python version uses high-level libraries such as `pandas`, `sklearn`, and `matplotlib` for data manipulation, machine learning, and plotting. The C version utilizes `libsvm` for SVM operations and requires manual data handling and result plotting.

### Installation

#### Prerequisites

- **For Python:**
  - Python 3.x
  - pandas
  - scikit-learn
  - matplotlib

- **For C:**
  - GCC Compiler
  - Make
  - libsvm

#### Python Setup

1. **Install Python:** Ensure Python 3.x is installed on your system. If not, download and install it from [python.org](https://www.python.org/downloads/).

2. **Install Required Libraries:**
   ```bash
   pip install pandas scikit-learn matplotlib
   ```

#### C Setup

1. **Install GCC and Make:** Ensure GCC and Make are installed on your system. These are typically available by default on Linux and MacOS. For Windows, consider using MinGW or Cygwin.

2. **Install libsvm:**
   - Clone the libsvm repository:
     ```bash
     git clone https://github.com/cjlin1/libsvm.git
     ```
   - Compile the library:
     ```bash
     cd libsvm
     make
     ```

### Usage

#### Python

1. **Run the Python Script:**
   - Navigate to the directory containing your Python script.
   - Execute the script:
     ```bash
     python your_script.py
     ```

#### C

1. **Compile the C Program:**
   - Ensure your C files and the libsvm source are in the same directory.
   - Compile using:
     ```bash
     gcc -o svm_program your_program.c -L./libsvm -lsvm
     ```

2. **Run the Compiled Program:**
   ```bash
   ./svm_program
   ```

### File Structure

- **Python:**
  - `your_script.py`: Main Python script for training and predicting using One-Class SVM.

- **C:**
  - `your_program.c`: Main C program file for SVM operations.
  - `/libsvm`: Directory containing the libsvm source code and library files.

### Notes

- Ensure that the CSV files are in the correct format as expected by each script. Python scripts handle CSV files directly through pandas, whereas C programs require manual parsing.
- The C implementation may need modifications for specific use cases, especially for data handling and plotting which are not as straightforward as in Python.


--- 

