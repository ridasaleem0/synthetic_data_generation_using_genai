# Synthetic Data Generation and Evaluation 

## Overview

This Python script demonstrates the process of generating synthetic data using multiple state of the art machine learning synthesizers available in the SDV (Synthetic Data Vault) library. The script experiments with `GaussianCopulaSynthesizer`, `CTGANSynthesizer`, `CopulaGANSynthesizer`, and `TVAESynthesizer` to create synthetic datasets based on the characteristics and patterns observed in the provided `mle_test_data.csv` file. Each synthesizer is evaluated against the original data to assess its effectiveness in replicating key statistical properties and relational dependencies.

## Prerequisites

Before running the script, ensure you have the following installed:
- Python 3.x
- Required Python packages (`pandas`, `sdv`, etc.). Install them using:
  ```
  pip install pandas sdv
  ```

## Steps to Run the Script

1. **Download the Python file:**
   ```
   cd <folder_name>
   ```

2. **Download the Data:**
   Place your `mle_test_data.csv` file in the root directory of the folder.

3. **Run the Script:**
   Execute the Python script `generate_synthetic_data.py`:
   ```
   python generate_synthetic_data.py
   ```

4. **View Results:**
   - The script will generate synthetic data using each synthesizer and evaluate it against the real data.
   - It will display diagnostic checks, quality evaluation reports, and visual comparisons between real and synthetic data for each synthesizer.

## Script Details

- **Synthesizers Used:** The script uses the following synthesizers from SDV:
  - `GaussianCopulaSynthesizer`
  - `CTGANSynthesizer`
  - `CopulaGANSynthesizer`
  - `TVAESynthesizer`
  
- **Evaluation Criteria:** Synthetic data generated by each synthesizer is evaluated based on:
  - Temporal coherence (e.g., policy dates order).
  - Statistical comparison (e.g., distribution of `sum_insured`, `square_foot_area`, `num_stories`).
  - Row-level coherence (e.g., correspondence between `construction_description` and `oed_construction_code`).

- **Visualization:** The script includes visualizations to compare distributions and correlations between real and synthetic data for each synthesizer.

## Output

- The script saves the generated synthetic data for each synthesizer to separate CSV files (`synthetic_data_<synthesizer_name>.csv`) in the root directory.
- Evaluation results and visualizations are displayed during script execution for each synthesizer.

## Best Performing Model
Rigorous training and experimentation ensured that the GaussianCopulaSynthesizer was effective in
generating synthetic data that closely resembled the statistical patterns and dependencies observed in the
mle_test_data.csv. It excelled in preserving the marginal distributions of individual variables while
capturing the linear correlations between them using Gaussian copulas. This approach ensured that the
synthetic data maintained the integrity of the original data structure, making it suitable for scenarios
where maintaining data coherence and dependency relationships is critical.
1. The temporal coherence of the synthetic data was assessed by verifying that the policy_end_date
consistently follows the policy_start_date, thereby maintaining logical consistency in temporal ordering.
2. A thorough comparison of statistical patterns was performed, focusing on key numerical variables such as
sum_insured and square_foot_area. This comparison confirmed that the synthetic data closely mirrored the
distributional characteristics observed in the original test data, indicating robustness in replicating
statistical properties
3. the coherence within each row of synthetic data was scrutinized, particularly concerning the alignment
between construction_description and oed_construction_code.

### Data Validity and Structure
The Gaussian Copula Synthesizer demonstrated the highest performance based on the evaluation metrics.

- Data Validity Score: 99.62%
- Data Structure Score: 100%
- Overall Validity and Structure Score: 99.81%
This combined score underscores the high fidelity of the synthetic data in terms of both individual data points and their interrelationships.

- Column Shapes Score: 95.39%
- Column Pair Trends Score: 84.53%
- Overall Quality Score: 89.96%
This comprehensive score reflects the overall similarity between the real and synthetic data in terms of both column distributions and pairwise relationships.

Detailed Column Shapes Report
- policy_start_date: 96.81%
- policy_end_date: 97.21%
- sum_insured: 96.96%
- construction_description: 95.02%
- year_built: 90.23%
- num_stories: 98.07%
- square_foot_area: 89.99%
- oed_construction_code: 98.80%

Number of Violations: 0
This indicates that all constraints (e.g., policy_end_date being after policy_start_date) have been perfectly adhered to in the synthetic data.

## Additional Notes

- Adjust parameters in the script (e.g., epochs for CTGANSynthesizer) as needed for your specific dataset and requirements.
- Further fine-tuning model parameters, loss functions, and training strategies would allow customization to match more specific data distribution nuances:
  - Adjusting batch sizes, learning rates, and optimizer settings can influence how well the model replicates data distributions.
  - Tailoring model architectures to handle features like skewed distributions, multi-modal data, or rare events improves fidelity.
- Moreover, developing robust evaluation metrics to assess synthetic data against real data benchmarks would ensure distributional fidelity:
  - Statistical similarity metrics (e.g., Kolmogorov-Smirnov test, Jensen-Shannon divergence) quantify how closely synthetic data matches original distributions.
  - Visualizations and exploratory data analysis help intuitively verify the replication of distributional patterns.
- Iteratively refining models based on validation results and domain-specific insights can enhance the accuracy of synthetic data distribution.
- Ensure proper handling of warnings and error messages during script execution.
- This script is not yet pushed to GitHub.

---
