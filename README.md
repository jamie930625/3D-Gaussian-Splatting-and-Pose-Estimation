# DLCV-Fall-2025-HW4

Please click [this link](https://docs.google.com/presentation/d/1ZnvPv5h4_8zWKe4LR_4Hv4UmmFxmTzMJLHJEhyOoNhY/edit?usp=sharing) to view the slides of HW4

# Usage

To start working on this assignment, you should clone this repository into your local machine by using the following command.
    
    git clone https://github.com/DLCV-Fall-2024/dlcv-fall-2024-hw4-<username>.git


Note that you should replace `<username>` with your own GitHub username.

---

### 1. Repository & Structure

1.  Please clone the official **DUSt3R** repository from:
    `https://github.com/naver/dust3r`

2.  Ensure the cloned repository is placed according to the required directory structure:
    <img width="418" height="213" alt="image" src="https://github.com/user-attachments/assets/3531d27e-a61c-4b29-9adf-09faa742b714" />

---

### 2. Required Modifications

The codebase has been largely pre-implemented by the TAs. Your primary task is limited to the following files:

* **`dust3r_inference.py`**: Modify the core logic to correctly handle different input types.
* **Bash Scripts**: Adjust the relevant execution scripts (e.g., `hw4_1_1.sh`, `hw4_1_2.sh`) to ensure correct input parsing and execution for both wide-baseline and interpolated inputs.

---

### 3. Reference Scripts

You have reference scripts available for debugging and testing your implementation:

* **Wide-Baseline Input:** Refer to **`bash scripts/dust3r/public/save_dust3r_pair.sh`** for saving predictions from **wide-baseline pair inputs**.
* **Testing Metrics:** Test your output and check metrics using **`bash scripts/dust3r/public/run_check_metrics.sh`**.
* **Interpolated Input:** A similar script (check your provided files, likely named **`save_dust3r_interpolated.sh`** or similar) handles the saving for **interpolated sequence inputs**. Use the logic from the wide-baseline script as a guide.

# Data
please download `dataset` from the link below
[https://drive.google.com/drive/folders/1QnxUnuygh6d9zIDGS8y83spoaugLG5Np?usp=sharing](https://drive.google.com/drive/folders/1QnxUnuygh6d9zIDGS8y83spoaugLG5Np?usp=sharing)


# Submission Rules
### Deadline
2025/12/2 (Tue.) 23:59

### Packages
This homework requires **Python 3.10** and specific packages for two distinct models. Due to potential dependency conflicts, you are **required to set up two separate virtual environments** (e.g., using `conda` or `venv`).

---

### Environment 1: Problem 1 (DUSt3R)

* **Setup:** Follow the specific setup and package requirements outlined in the **DUSt3R repository**.
* **Purpose:** Used for all tasks involving stereo matching and pose estimation on wide-baseline and interpolated image pairs.

### Environment 2: Problem 2 (InstantSplat)

* **Setup:** Follow the specific setup and package requirements outlined in the **InstantSplat repository**.
* **Purpose:** Used for all tasks involving 3D Gaussian Splatting and novel view synthesis.

---

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

# Q&A
If you have any problems related to HW4, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under HW4 discussion section on NTU COOL.
