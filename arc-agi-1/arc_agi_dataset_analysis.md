# ARC-AGI-1 Dataset Analysis Report

This report provides a comprehensive analysis of the ARC-AGI-1 dataset, including both training and evaluation sets.

## Dataset Overview

| Dataset | Total Tasks | Total Train Examples | Total Test Examples |
|---------|-------------|---------------------|--------------------|
| Training | 400 | 1302 | 416 |
| Evaluation | 400 | 1363 | 419 |
| **Total** | **800** | **2665** | **835** |

## Training Dataset Analysis

### Training Input Matrix Size Statistics

**Total matrices analyzed:** 1718

| Dimension | Min | Max | Mean | Std Dev |
|-----------|-----|-----|------|---------|
| Height | 1.0 | 30.0 | 10.40 | 6.08 |
| Width | 2.0 | 30.0 | 10.84 | 6.15 |
| Area | 4.0 | 900.0 | 144.92 | 164.29 |

**Most Common Matrix Sizes (Height × Width):**
1. 10×10: 270 matrices
2. 3×3: 197 matrices
3. 9×9: 85 matrices
4. 5×5: 54 matrices
5. 12×12: 48 matrices
6. 15×15: 43 matrices
7. 7×7: 39 matrices
8. 13×13: 38 matrices
9. 11×11: 38 matrices
10. 20×20: 35 matrices

### Training Output Matrix Size Statistics

**Total matrices analyzed:** 1718

| Dimension | Min | Max | Mean | Std Dev |
|-----------|-----|-----|------|---------|
| Height | 1.0 | 30.0 | 8.88 | 5.84 |
| Width | 1.0 | 30.0 | 9.22 | 6.06 |
| Area | 1.0 | 900.0 | 113.07 | 146.70 |

**Most Common Matrix Sizes (Height × Width):**
1. 10×10: 244 matrices
2. 3×3: 239 matrices
3. 9×9: 82 matrices
4. 4×4: 69 matrices
5. 6×6: 64 matrices
6. 5×5: 50 matrices
7. 15×15: 46 matrices
8. 12×12: 45 matrices
9. 1×1: 42 matrices
10. 7×7: 38 matrices

### Training Input Color Statistics

**Total pixels analyzed:** 248,975
**Unique colors found:** 10

**Color Distribution:**
| Color | Count | Percentage |
|-------|-------|-----------|
| 0 | 154,291 | 61.97% |
| 1 | 17,950 | 7.21% |
| 8 | 16,091 | 6.46% |
| 2 | 13,800 | 5.54% |
| 3 | 13,167 | 5.29% |
| 5 | 11,938 | 4.79% |
| 4 | 10,922 | 4.39% |
| 6 | 4,762 | 1.91% |
| 7 | 3,046 | 1.22% |
| 9 | 3,008 | 1.21% |

### Training Output Color Statistics

**Total pixels analyzed:** 194,260
**Unique colors found:** 10

**Color Distribution:**
| Color | Count | Percentage |
|-------|-------|-----------|
| 0 | 105,453 | 54.28% |
| 3 | 15,957 | 8.21% |
| 1 | 15,465 | 7.96% |
| 2 | 13,431 | 6.91% |
| 8 | 12,752 | 6.56% |
| 4 | 11,479 | 5.91% |
| 5 | 9,112 | 4.69% |
| 6 | 4,967 | 2.56% |
| 7 | 3,281 | 1.69% |
| 9 | 2,363 | 1.22% |

### Training Input Matrix Color Diversity

- Matrices with **one color**: 13 (0.76%)
- Matrices with **more than one color**: 1705 (99.24%)

### Training Output Matrix Color Diversity

- Matrices with **one color**: 69 (4.02%)
- Matrices with **more than one color**: 1649 (95.98%)

## Evaluation Dataset Analysis

### Evaluation Input Matrix Size Statistics

**Total matrices analyzed:** 1782

| Dimension | Min | Max | Mean | Std Dev |
|-----------|-----|-----|------|---------|
| Height | 1.0 | 30.0 | 13.23 | 7.22 |
| Width | 2.0 | 30.0 | 13.70 | 7.17 |
| Area | 4.0 | 900.0 | 226.92 | 226.01 |

**Most Common Matrix Sizes (Height × Width):**
1. 10×10: 157 matrices
2. 3×3: 104 matrices
3. 15×15: 84 matrices
4. 30×30: 74 matrices
5. 13×13: 48 matrices
6. 12×12: 46 matrices
7. 16×16: 42 matrices
8. 6×6: 38 matrices
9. 20×20: 37 matrices
10. 4×4: 37 matrices

### Evaluation Output Matrix Size Statistics

**Total matrices analyzed:** 1782

| Dimension | Min | Max | Mean | Std Dev |
|-----------|-----|-----|------|---------|
| Height | 1.0 | 30.0 | 11.71 | 7.23 |
| Width | 1.0 | 30.0 | 12.18 | 7.24 |
| Area | 1.0 | 900.0 | 189.67 | 216.84 |

**Most Common Matrix Sizes (Height × Width):**
1. 10×10: 162 matrices
2. 3×3: 93 matrices
3. 15×15: 75 matrices
4. 9×9: 71 matrices
5. 30×30: 60 matrices
6. 6×6: 53 matrices
7. 5×5: 52 matrices
8. 4×4: 50 matrices
9. 12×12: 50 matrices
10. 13×13: 41 matrices

### Evaluation Input Color Statistics

**Total pixels analyzed:** 404,366
**Unique colors found:** 10

**Color Distribution:**
| Color | Count | Percentage |
|-------|-------|-----------|
| 0 | 237,571 | 58.75% |
| 1 | 34,788 | 8.60% |
| 8 | 28,608 | 7.07% |
| 2 | 20,176 | 4.99% |
| 3 | 20,031 | 4.95% |
| 5 | 19,262 | 4.76% |
| 4 | 15,639 | 3.87% |
| 6 | 10,743 | 2.66% |
| 7 | 9,593 | 2.37% |
| 9 | 7,955 | 1.97% |

### Evaluation Output Color Statistics

**Total pixels analyzed:** 337,985
**Unique colors found:** 10

**Color Distribution:**
| Color | Count | Percentage |
|-------|-------|-----------|
| 0 | 179,481 | 53.10% |
| 1 | 32,323 | 9.56% |
| 8 | 25,445 | 7.53% |
| 2 | 23,921 | 7.08% |
| 3 | 21,750 | 6.44% |
| 4 | 17,217 | 5.09% |
| 5 | 14,854 | 4.39% |
| 6 | 9,771 | 2.89% |
| 7 | 9,133 | 2.70% |
| 9 | 4,090 | 1.21% |

### Evaluation Input Matrix Color Diversity

- Matrices with **one color**: 10 (0.56%)
- Matrices with **more than one color**: 1772 (99.44%)

### Evaluation Output Matrix Color Diversity

- Matrices with **one color**: 32 (1.80%)
- Matrices with **more than one color**: 1750 (98.20%)

## Combined Dataset Statistics

### Combined Input Matrix Size Statistics

**Total matrices analyzed:** 3500

| Dimension | Min | Max | Mean | Std Dev |
|-----------|-----|-----|------|---------|
| Height | 1.0 | 30.0 | 11.84 | 6.83 |
| Width | 2.0 | 30.0 | 12.30 | 6.84 |
| Area | 4.0 | 900.0 | 186.67 | 202.32 |

**Most Common Matrix Sizes (Height × Width):**
1. 10×10: 427 matrices
2. 3×3: 301 matrices
3. 15×15: 127 matrices
4. 9×9: 121 matrices
5. 12×12: 94 matrices
6. 30×30: 92 matrices
7. 13×13: 86 matrices
8. 5×5: 83 matrices
9. 7×7: 76 matrices
10. 20×20: 72 matrices

### Combined Output Matrix Size Statistics

**Total matrices analyzed:** 3500

| Dimension | Min | Max | Mean | Std Dev |
|-----------|-----|-----|------|---------|
| Height | 1.0 | 30.0 | 10.32 | 6.74 |
| Width | 1.0 | 30.0 | 10.73 | 6.85 |
| Area | 1.0 | 900.0 | 152.07 | 189.66 |

**Most Common Matrix Sizes (Height × Width):**
1. 10×10: 406 matrices
2. 3×3: 332 matrices
3. 9×9: 153 matrices
4. 15×15: 121 matrices
5. 4×4: 119 matrices
6. 6×6: 117 matrices
7. 5×5: 102 matrices
8. 12×12: 95 matrices
9. 30×30: 75 matrices
10. 7×7: 73 matrices

### Combined Input Color Statistics

**Total pixels analyzed:** 653,341
**Unique colors found:** 10

**Color Distribution:**
| Color | Count | Percentage |
|-------|-------|-----------|
| 0 | 391,862 | 59.98% |
| 1 | 52,738 | 8.07% |
| 8 | 44,699 | 6.84% |
| 2 | 33,976 | 5.20% |
| 3 | 33,198 | 5.08% |
| 5 | 31,200 | 4.78% |
| 4 | 26,561 | 4.07% |
| 6 | 15,505 | 2.37% |
| 7 | 12,639 | 1.93% |
| 9 | 10,963 | 1.68% |

### Combined Output Color Statistics

**Total pixels analyzed:** 532,245
**Unique colors found:** 10

**Color Distribution:**
| Color | Count | Percentage |
|-------|-------|-----------|
| 0 | 284,934 | 53.53% |
| 1 | 47,788 | 8.98% |
| 8 | 38,197 | 7.18% |
| 3 | 37,707 | 7.08% |
| 2 | 37,352 | 7.02% |
| 4 | 28,696 | 5.39% |
| 5 | 23,966 | 4.50% |
| 6 | 14,738 | 2.77% |
| 7 | 12,414 | 2.33% |
| 9 | 6,453 | 1.21% |

### Combined Input Matrix Color Diversity

- Matrices with **one color**: 23 (0.66%)
- Matrices with **more than one color**: 3477 (99.34%)

### Combined Output Matrix Color Diversity

- Matrices with **one color**: 101 (2.89%)
- Matrices with **more than one color**: 3399 (97.11%)

## Task Complexity Analysis

### Training Examples per Task

| Dataset | Min | Max | Mean | Std Dev |
|---------|-----|-----|------|---------|
| Training | 2.0 | 10.0 | 3.25 | 0.96 |
| Evaluation | 2.0 | 7.0 | 3.41 | 0.98 |

## Key Findings

### Matrix Sizes
- Input matrices range from the smallest to largest observed sizes
- Output matrices show similar size distributions
- Most tasks involve relatively small matrices (typical sizes under 30×30)

### Color Usage
- The dataset uses colors 0-9 (10 total colors)
- Color 0 (typically background) is most frequent
- Color distribution varies between training and evaluation sets

### Task Structure
- Training set: 400 tasks
- Evaluation set: 400 tasks
- Each task contains multiple input-output examples for learning the pattern

---
*Report generated by ARC-AGI-1 Dataset Analyzer*
