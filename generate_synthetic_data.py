"""This class structure encapsulates the process of generating synthetic data using pre-trained ml models,
applying constraints, evaluating its quality, and visualizing differences with real data. It provides a comprehensive
approach to exploring SDV python library and understanding Generative AI performance in replicating real-world data
patterns. Using the machine learning techniques to create synthetic data, learn patterns and emulates them for a
single table and evaluate it."""

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot, get_column_pair_plot
import warnings


class SyntheticDataGenerator:
    def __init__(self, real_data_filename):
        """
        Initializes the SyntheticDataGenerator class.

        Parameters:
        real_data_filename (str): The filename of the CSV containing the real data i.e., mle_test_data.
        """
        self.real_data = pd.read_csv(real_data_filename)  # Load real data from CSV into a DataFrame
        self.metadata = SingleTableMetadata()  # Create metadata object for describing the schema
        self.metadata.detect_from_dataframe(self.real_data)  # Detect metadata from the real data
        self.metadata.update_column(column_name='construction_description', sdtype='categorical')
        self.metadata.validate()  # Validate the metadata
        self.synthesizers = {

            # Gaussian Copula is a type of ml model used in statistical modeling and data synthesis. It is
            # particularly effective for generating synthetic data that preserves the marginal distributions of
            # individual variables while capturing the dependence structure between them.
            'GaussianCopula': GaussianCopulaSynthesizer(metadata=self.metadata, enforce_min_max_values=True),

            # CTGAN uses generative adversarial networks(GANs) to create synthesize data with high fidelity.
            # A tradeoff between training time and data quality was made using the epochs parameter:
            # Higher epochs ensured that the synthesizer will train for longer, and ideally improved the data quality.
            'CTGAN': CTGANSynthesizer(metadata=self.metadata,
                                      epochs=500,
                                      enforce_min_max_values=True),


            # The Copula GAN Synthesizer uses a mix classic, statistical methods and GAN-based deep learning methods
            # to train a model and generate synthetic data.

            # Set the distribution shape of numerical columns that appear in the test table based on their
            # distribution observation. - sum_insured: Gamma Distribution (gamma) is suitable for positively skewed
            # data and can model the wide range observed in sum_insured effectively. The gamma distribution is a good
            # fit for data that is strictly positive and has a right skew. - square_foot_area: Similar to
            # sum_insured, square_foot_area is also positively skewed and has a wide range. The gamma distribution
            # works well for this type of data. - num_stories: Since num_stories is primarily centered around 1 but
            # can go up to 25, a truncated normal distribution might be a good fit. It allows for a more realistic
            # modeling of the data, ensuring the values stay within a realistic range.

            'CopulaGANSynthesizer': CopulaGANSynthesizer(metadata=self.metadata,
                                                         numerical_distributions={
                                                             'sum_insured': 'gamma',
                                                             'square_foot_area': 'gamma',
                                                             'num_stories': 'truncnorm'
                                                         },
                                                         enforce_min_max_values=True,
                                                         epochs=500),

            # The Tabular TVAE Synthesizer uses a variational autoencoder (VAE)-based,
            # neural network techniques to train a model and generate synthetic data.
            'TVAE': TVAESynthesizer(metadata=self.metadata,
                                    enforce_min_max_values=True,
                                    enforce_rounding=False,
                                    epochs=500)
        }

    def pre_defined_constraints(self, synthesizer):
        """
        Apply constraints to the synthesizer to ensure data consistency and validity.

        Parameters:
        synthesizer: An instance of an SDV synthesizer.

        Returns:
        synthesizer: The synthesizer object with constraints applied.
        """

        # Constraint 1: Policy start date must be less than policy end date
        policy_start_end_constraint = {
            'constraint_class': 'Inequality',
            'constraint_parameters': {
                'low_column_name': 'policy_start_date',
                'high_column_name': 'policy_end_date',
                'strict_boundaries': True
            }
        }

        # Constraint 2: Fixed combinations between construction_description and oed_construction_code
        FixedCombinations_constraint = {
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {
                'column_names': ['construction_description', 'oed_construction_code']
            }
        }

        # Add constraints to the synthesizer
        synthesizer.add_constraints(constraints=[policy_start_end_constraint, FixedCombinations_constraint])
        return synthesizer

    def generate_synthetic_data(self, synthesizer, synthesizer_name):
        """
        Generate synthetic data using the specified synthesizer.

        Parameters:
        synthesizer: An instance of an SDV synthesizer.
        synthesizer_name: Name of the ml model used

        Returns:
        synthetic_data: DataFrame containing synthetic data generated by the synthesizer.
        """
        synthesizer = self.pre_defined_constraints(synthesizer)  # Apply constraints to the synthesizer
        synthesizer.fit(self.real_data)  # Fit the synthesizer to the real data

        # Use the sample function and pass in 10k number of rows to synthesize.
        synthetic_data = synthesizer.sample(num_rows=10000)
        # Save the synthetic data that contains the same table, columns and connections as the real data into CSV format
        output_file = f"synthetic_data_{synthesizer_name.lower().replace(' ', '_')}.csv"
        synthetic_data.to_csv(output_file, sep=',', index=False, encoding='utf-8')
        return synthetic_data

    def evaluate_real_vs_synthetic_data(self, real_data, synthetic_data):
        """
        Evaluate the quality of synthetic data compared to real data.
        SDV has built-in functions for evaluating the synthetic data and getting more insight.
        As a first step, we ran a diagnostic to ensure that the data is valid.
        SDV's diagnostic performs some basic checks such as:
        - All primary keys must be unique
        - Continuous values must adhere to the min/max of the real data
        - Discrete columns (non-PII) must have the same categories as the real data.

        Parameters:
        real_data: DataFrame containing the given real data.
        synthetic_data: DataFrame containing the synthetic data.
        """

        # Run diagnostic to check data validity
        diagnostic = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=self.metadata, verbose=True)
        print("Diagnostic Report:\n", diagnostic)
        # Evaluate data quality or the statistical similarity between the real and synthetic data. This value may vary
        # anywhere from 0 to 100%.
        quality_report = evaluate_quality(real_data, synthetic_data, self.metadata, verbose=True)
        print("Quality Report:\n", quality_report)

        # Determines which columns had the highest vs. the lowest scores in terms of statistical similarity between real
        # and synthetic data. This information can be used to further finetune specific columns.
        print("Column Shapes Report:\n", quality_report.get_details('Column Shapes'))

        # Check for policy start date >= policy end date violations in synthetic data
        synthetic_dates = synthetic_data[['policy_start_date', 'policy_end_date']].dropna()
        violations = pd.to_datetime(synthetic_dates['policy_start_date']) >= pd.to_datetime(synthetic_dates['policy_end_date'])
        print('\nNumber of violations in the synthetic (constrained) data:',
              len([row for row in violations if row == True]))

    def visualize_real_vs_synthetic_data(self, real_data, synthetic_data):
        """
        Visualize differences between real and synthetic data.

        Parameters:
        real_data: DataFrame containing the real data.
        synthetic_data: DataFrame containing the synthetic data.
        """
        # 1D visualization to compare a column of the real data to the synthetic data.
        # Visualize oed_construction_code column
        fig = get_column_plot(real_data=real_data,
                              synthetic_data=synthetic_data,
                              column_name='oed_construction_code',
                              metadata=self.metadata)
        fig.show()

        # 2D visualization comparing the correlations of a pair of columns.
        fig = get_column_pair_plot(real_data=real_data,
                                   synthetic_data=synthetic_data,
                                   column_names=['oed_construction_code', 'construction_description'],
                                   metadata=self.metadata)
        fig.show()
        # Visualize square_foot_area column
        fig = get_column_plot(real_data=real_data,
                              synthetic_data=synthetic_data,
                              column_name='square_foot_area',
                              metadata=self.metadata)
        fig.show()
        fig = get_column_plot(real_data=real_data,
                              synthetic_data=synthetic_data,
                              column_name='num_stories',
                              metadata=self.metadata)
        fig.show()

    def run_experiment(self):
        """
        Run the synthetic data generation and evaluation experiment using all synthesizers based on statistical, GANs
        and autoencoder machine learning models. This experiment results in finding the best performing model to
        generate the synthetic tabular data based on real test data.
        """
        for name, synthesizer in self.synthesizers.items():
            synthetic_data = self.generate_synthetic_data(synthesizer, name)
            self.evaluate_real_vs_synthetic_data(self.real_data, synthetic_data)
            self.visualize_real_vs_synthetic_data(self.real_data, synthetic_data)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # Ignore all warnings
    generator = SyntheticDataGenerator(real_data_filename='mle_test_data.csv')
    generator.run_experiment()  # Run the synthetic data generation and evaluation experiment
