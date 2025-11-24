# realm_task_3_3_implementation_xai

## General Task Description

Components developed in Task 3.3 aim to implement agnostic XAI techniques on top of AI models that are used for various tasks such as classification or segmentation. We aim to implement two XAI techniques per Use Case - that would be selected dynamically from the Fuzzy system based on User's Input (sensitivity value coming from the RIANA dashboard), implement bias and fairness metrics (as agreed [here](https://maastrichtuniversity.sharepoint.com/:w:/r/sites/FSE-REALM/_layouts/15/Doc.aspx?sourcedoc=%7B9EDAE561-2787-42D1-BBB8-C9320C0B1F25%7D&file=Report%20on%20Bias%20and%20Fairness%20Metrics%20%5BTask%203.3%5D.docx&action=default&mobileredirect=true)) based on model outputs and extract outputs in a digestible manner (images, metrics, etc.).

This component, no matter the Use Case, expects as input:
- Sensitivity value (RIANA dashboard)
- Trained model (AI Orchestrator)
- Compatible dataset (AI Orchestrator)

This component, no matter the Use Case, returns as output:
- XAI methodology output (depending on the Use Case - image or json file)
- Fairness and Bias results (depending on the Use Case - json file or nothing if we are talking for images)


## COPD Triage Prediction with Explainability (Use Case 5)

This project provides tools for analyzing and interpreting the predictions made by the COPowereD model, which predicts the need for medical attention in COPD patients based on self-measured ambulatory parameters.  
It uses SHAP and LIME explainability methods to identify key features influencing the model's predictions.  
Additionally, it includes fairness/bias analysis to detect potential biases across demographic groups (Age and Gender).

## Project Overview

The COPowereD model takes clinical biomedical data and predicts whether a COPD patient needs medical attention. This project adds an explainability layer, allowing users to understand why the model makes certain predictions and assess potential biases.

**IMPORTANT**: The model is a Gradient Boosted Tree classifier trained on tabular clinical data. The explainability analysis uses model-agnostic techniques (SHAP for higher sensitivity (≥ 0.5), LIME for lower sensitivity (< 0.5)) to provide interpretable insights.

Key Components:
1. Input Data: CSV file containing clinical biomedical data for COPD patients with features including demographics, baseline vitals, and current symptoms. More details in the [Data Structure section](#data-structure) below.
2. Model: Gradient Boosted Tree Pipeline (GBTP) for binary classification (`needMedicalAttention` vs `notNeedMedicalAttention`). Model details available in the provided model card.
3. Fairness/Bias Analysis: Performed using the [fairness_bias_analysis.py](./fairness_bias_analysis.py) script. Evaluates equalized odds and demographic parity across Age and Gender groups. This script can be executed independently and as part of the Kubeflow pipeline component.
4. Explainability Analysis: Performed using the [explainer.py](./explainer.py) script with dynamic method selection based on sensitivity value. Again, this script can be executed independently and as part of the Kubeflow pipeline component.
5. Visualizations: Generated using [fairness_bias_visualization.py](./fairness_bias_visualization.py) and [explainer_visualization.py](./explainer_visualization.py) scripts.


## Getting Started

### Prerequisites
- Python 3.13 or higher
- Required Python packages (installed via `pip install -r requirements.txt`, can be found in [requirements.txt](./requirements.txt))

### Data Structure

The input data is expected to be in CSV format with the following structure (features and label). The label column can take two values, 0 and 1 (label: 0 = No medical attention needed, 1 = Medical attention needed).

| id | Gender | Age | Height | Weight | BMI | b_COPD | b_HeartRate | b_SPO2 | s_Worsening | s_Breath | s_Cough | s_Sputum | c_HeartRate | c_SPO2 | label |
|----|--------|-----|--------|--------|-----|--------|-------------|--------|-------------|----------|---------|----------|-------------|--------|-------|
| 0  | 1      | 65  | 175    | 80     | 26.1| 2      | 75          | 95     | 1           | 2        | 2       | 1        | 82          | 93     | 1     |
| 1  | 0      | 58  | 162    | 68     | 25.9| 1      | 68          | 97     | 0           | 1        | 1       | 0        | 70          | 96     | 0     |

### Running the COPowereD Model

The COPowereD model is accessible via a REST API. Use the following command to make predictions:

```bash
curl -X POST https://canalytics.comunicare.io/api/prediction \
  -H "Content-Type: application/json" \
  -d '{
  "methods": [
    {
      "project": "BpcoTriagingBinary",
      "algo": "MLGBTPipeline",
      "version": "0.0.1"
    }
  ],
  "observations": [
    {
      "subject": {
        "reference": "patient_001"
      },
      "component": [
        {
          "valueQuantity": {"value": 1},
          "code": {
            "coding": [{"code": "46098-0", "display": "sex", "system": "http://loinc.org"}]
          }
        },
        {
          "valueQuantity": {"value": 65},
          "code": {
            "coding": [{"code": "63900-5", "display": "Current age or age at death", "system": "http://loinc.org"}]
          }
        },
        {
          "valueQuantity": {"value": 175},
          "code": {
            "coding": [{"code": "8302-2", "display": "Body height", "system": "http://loinc.org"}]
          },
          "instant": "baseline"
        },
        {
          "valueQuantity": {"value": 80},
          "code": {
            "coding": [{"code": "3141-9", "display": "Body Weight", "system": "http://loinc.org"}]
          },
          "instant": "baseline"
        },
        {
          "valueQuantity": {"value": 26.1},
          "code": {
            "coding": [{"code": "39156-5", "display": "Body mass index", "system": "http://loinc.org"}]
          },
          "instant": "baseline"
        },
        {
          "valueQuantity": {"value": 2},
          "code": {
            "coding": [{"code": "13645005", "display": "COPD Gold stage", "system": "http://snomed.info/sct"}]
          },
          "instant": "baseline"
        },
        {
          "valueQuantity": {"value": 75},
          "code": {
            "coding": [{"code": "8867-4", "display": "Heart rate baseline", "system": "http://loinc.org"}]
          },
          "instant": "baseline"
        },
        {
          "valueQuantity": {"value": 95},
          "code": {
            "coding": [{"code": "20564-1", "display": "Oxygen saturation baseline", "system": "http://loinc.org"}]
          },
          "instant": "baseline"
        },
        {
          "valueQuantity": {"value": 1},
          "code": {
            "coding": [{"code": "275723000", "display": "Deteriorating condition", "system": "http://snomed.info/sct"}]
          }
        },
        {
          "valueQuantity": {"value": 2},
          "code": {
            "coding": [{"code": "267036007", "display": "Dyspnea", "system": "http://snomed.info/sct"}]
          }
        },
        {
          "valueQuantity": {"value": 2},
          "code": {
            "coding": [{"code": "49727002", "display": "Cough", "system": "http://snomed.info/sct"}]
          }
        },
        {
          "valueQuantity": {"value": 1},
          "code": {
            "coding": [{"code": "248595008_365445003", "display": "Sputum : Volume and Color", "system": "http://snomed.info/sct"}]
          }
        },
        {
          "valueQuantity": {"value": 82},
          "code": {
            "coding": [{"code": "8867-4", "display": "Heart rate", "system": "http://loinc.org"}]
          }
        },
        {
          "valueQuantity": {"value": 93},
          "code": {
            "coding": [{"code": "20564-1", "display": "Oxygen saturation", "system": "http://loinc.org"}]
          }
        }
      ]
    }
  ]
}'
```

An example response is shown below:

```json
{
  "success": true,
  "message": null,
  "data": [
    {
      "subject": {
        "reference": "patient_001"
      },
      "identifier": [
        {
          "value": "Risk_Copd_Triaging_Binary",
          "system": "http://comunicare.io"
        }
      ],
      "prediction": [
        {
          "outcome": {
            "coding": [
              {
                "code": "needMedicalAttention",
                "display": "Need for medical attention",
                "system": "http://comunicare.io"
              }
            ]
          },
          "rationale": "GBTP",
          "probabilityDecimal": 0.5320359882519113,
          "order": 1,
          "version": "0.0.1"
        },
        {
          "outcome": {
            "coding": [
              {
                "code": "notNeedMedicalAttention",
                "display": "No need for medical attention",
                "system": "http://comunicare.io"
              }
            ]
          },
          "rationale": "GBTP",
          "probabilityDecimal": 0.46796401174808866,
          "order": 0,
          "version": "0.0.1"
        },
        {
          "outcome": {
            "coding": [
              {
                "code": "notNeedMedicalAttention",
                "display": "No need for medical attention",
                "system": "http://comunicare.io"
              }
            ]
          },
          "rationale": "summary",
          "probabilityDecimal": 0.46796401174808866,
          "order": 0
        },
        {
          "outcome": {
            "coding": [
              {
                "code": "needMedicalAttention",
                "display": "Need for medical attention",
                "system": "http://comunicare.io"
              }
            ]
          },
          "rationale": "summary",
          "probabilityDecimal": 0.5320359882519113,
          "order": 1
        }
      ],
      "statut": "final"
    }
  ]
}
```

The response includes:
- success: Boolean indicating if the API call was successful.
- message: Error message (null if successful).
- data: Array containing prediction results for each patient.
  - subject: Patient reference ID.
  - identifier: Model identification information.
  - prediction: Array of predictions with both GBTP model results and summary results.
    - outcome: The predicted class code (needMedicalAttention or notNeedMedicalAttention).
    - probabilityDecimal: Probability score for the predicted class.

### Analyses Execution

In order for the analyses to be executed:
- The provided dataset should be downloaded and stored.
- For an input CSV dataset, the explainability and fairness analyses can be executed as described in the next sections.

#### Fairness/Bias Analysis

The fairness and bias analysis can be executed independently using the following command:
```bash
python fairness_bias_analysis.py --tabular_data 'path/to/tabular_data.csv' --pred_target 'path/to/pred.csv' --output 'output/fairness_analysis.json'
```

Arguments:
- `--tabular_data`: Path to the input tabular data CSV file.
- `--pred_target`: Path to the model's predicted labels CSV file.
- `--output`: Path to save the output JSON file containing the fairness and bias metrics.

The predicted labels are expected to be in CSV format with the following structure. The label column can take two values, 0 and 1 (label: 0 = No medical attention needed, 1 = Medical attention needed).

| label |
|-------|
| 1     |
| 0     |

Visualization of the results can be done using the `fairness_bias_visualization.py` script:

```bash
python fairness_bias_visualization.py --analysis_results 'output/fairness_analysis.json' --output output
```

Arguments:
- `--analysis_results`: Path to the JSON file containing the fairness and bias metrics.
- `--output`: Directory to save the generated visualizations. Default is `output`.

#### Explainability Analysis
The explainability analysis can be executed independently using the following command:

```bash
python explainer.py --tabular_data 'path/to/tabular_data.csv' --sensitivity 0.7 --output output
```

Arguments:
- `--tabular_data`: Path to the input tabular data CSV file.
- `--sensitivity`: Sensitivity value (between 0 and 1) to determine the explainability technique to be used. Default is `0.7`.
  - sensitivity < 0.5: LIME analysis
  - sensitivity ≥ 0.5: SHAP analysis
- `--output`: Directory to save the output files (explanations). Default is `output`.

Visualization of the results can be done with the `explainer_visualization.py` script:
```bash
python explainer_visualization.py --analysis_results path/to/output/shap_detailed_results.pickle --output output --sensitivity 0.5
```

Arguments:
- `--analysis_results`: Path to the analysis results file (`pickle` for SHAP, `json` for LIME).
- `--output`: Directory to save the generated visualizations. Default is `output`.
- `--sensitivity`: Sensitivity value (between 0 and 1) indicating the explainability technique used. Default is `0.7`.
  - sensitivity < 0.5: LIME analysis
  - sensitivity ≥ 0.5: SHAP analysis

Alterantively, the analyses and visualizations can be executed as part of the Kubeflow pipeline component, as described in the [Kubeflow Pipeline Component](#kubeflow-pipeline-component) section below.


## JSON Output

### Fairness and Bias Analysis Output
The fairness and bias analysis outputs a JSON file, `fairness_bias_analysis.json`, with the following structure. The metrics are calculated based on two demographic attributes, `Age` and `Gender`.

```json
{
    "equalized_odds_metrics": {
        "Age": {
            "error_rates_by_group": {
                "70+": {
                    "false_positive_rate": 0.0
                },
                "55-69": {
                    "false_positive_rate": 0.5
                },
                "40-54": {
                    "false_positive_rate": 0.25
                }
            }
        },
        "Gender": {
            "error_rates_by_group": {
                "0": {
                    "false_positive_rate": 0.0
                },
                "1": {
                    "false_positive_rate": 0.4
                }
            }
        }
    },
    "demographic_parity_metrics": {
        "Age": {
            "prediction_rates_by_group": {
                "70+": 0.4444444444444444,
                "55-69": 0.2,
                "40-54": 0.3333333333333333
            }
        },
        "Gender": {
            "prediction_rates_by_group": {
                "0": 0.2727272727272727,
                "1": 0.4444444444444444
            }
        }
    }
}
```

### Explainability Analysis Output
The explainability analysis outputs can either refer to SHAP analysis or LIME explanations, depending on the sensitivity value provided as input.

#### SHAP Analysis Output (sensitivity ≥ 0.5)
SHAP analysis produces two files:
- `shap_analysis.json`: Global feature importance metrics
- `shap_detailed_results.pickle`: Complete SHAP values for all instances

`shap_analysis.json` structure:
```json
{
    "feature_importance": [
        0.2055,
        0.1406,
        0.0309,
        ...
    ],
    "shap_direction_mean": [
        0.1105,
        0.0906,
        -0.0309,
        ...
    ],
    "shap_std": [
        0.0105,
        0.0089,
        0.0047,
        ...
    ],
    "shap_values": [
        [0.2211, 0.1495, -0.0355, ...],
        [0.1, 0.0318, -0.0262, ...],
        ...
    ],
    "features": [
        "Gender",
        "Age",
        "b_HeartRate",
        ...
    ]
}
```

The `shap_detailed_results.pickle` contains the complete SHAP values object for all instances, exactly as extracted from the SHAP analysis process.

#### LIME Analysis Output (sensitivity < 0.5)
LIME analysis produces two files:
- `lime_analysis.json`: Global feature importance aggregated from local explanations
- `lime_detailed_results.json`: Per-instance feature weights

`lime_analysis.json` structure:
```json
{
    "c_SPO2": {
        "importance": 0.2231,
        "mean_weight": 0.1113,
        "std_weight": 0.0431
    },
    "s_Sputum": {
        "importance": 0.1554,
        "mean_weight": 0.0916,
        "std_weight": 0.0554
    },
    "Age": {
        "importance": 0.0209,
        "mean_weight": 0.0189,
        "std_weight": 0.0201
    },
    ...
}
```

`lime_detailed_results.json` structure (sample):
```json
[
    {
        "instance_idx": 0,
        "true_class": 1,
        "predicted_class": 1,
        "explained_class": 1,
        "c_SPO2": 0.1244,
        "s_Sputum": -0.0538,
        "Age": 0.0210,
        "Height": -0.0129,
        ...
    },
    {
        "instance_idx": 1,
        "true_class": 1,
        "predicted_class": 1,
        "explained_class": 1,
        "c_SPO2": -0.1218,
        "s_Sputum": 0.0570,
        ...
    }
]
```

## Visualizations Output

### Fairness and Bias Analysis Visualizations

Equalized Odds and Demographic Parity Plots for Age Demographic column:
![Kubeflow Pipeline](./media/age_fairness_bias.png)

Equalized Odds and Demographic Parity Plots for Gender Demographic column:
![Kubeflow Pipeline](./media/gender_fairness_bias.png)

### Explainability Analysis Visualizations
SHAP and LIME Explanations Plots
<p align="center">
  <img src="./media/shap_bar_plot.png" width="26%"/>
  <img src="./media/shap_beeswarm_plot.png" width="30%"/>
  <img src="./media/lime_barplot.png" width="41%"/>
</p>

## Understanding the Results

### Fairness and Bias Analysis Output
The fairness analysis produces:
- Equalized Odds Metrics: Measures whether error rates are similar across demographic groups.
- Demographic Parity Metrics: Ensures prediction rates are similar across demographic groups.
- Visualizations: Bar Charts visualizing the above results.

### Explainability Analysis Output
For SHAP:
- Global feature importance identifies which features most influence model predictions overall (averaged across all instances).
- `feature_importance`: Higher values = stronger average impact on predictions.
- `shap_direction_mean`: Indicates whether feature generally increases (positive) or decreases (negative) the probability of needing medical attention.
- `shap_std`: Indicates consistency - lower values mean the feature has similar impact across instances.
- `shap_values`: Instance-level SHAP values (rows = instances, columns = features).
- The Bar Plot shows global feature importance (mean absolute SHAP value). Higher bars indicate features with stronger average impact on predictions.
- `Beeswarm Plot`: Shows the distribution of SHAP values for each feature across all instances. Each dot represents one instance, colored by feature value (high=red, low=blue). Position indicates the SHAP value (impact on prediction).

For LIME:
- Global importance is derived from averaging local explanations across all instances.
- `importance`: Mean absolute weight indicates average magnitude of influence.
- `mean_weight`: Indicates average direction (positive/negative).
- `std_weight`: Indicates variability - higher values mean the feature's importance varies more across instances.
- Detailed results show which features were most important for each individual prediction, allowing for instance-level interpretation.
- Thge Bar Plot shows aggregated global feature importance from LIME local explanations. Features are ranked by their average absolute weight across all instances.


## Kubeflow Pipeline Component

The [copowered_pipeline_component.py](./kubeflow_component/copowered_pipeline_component.py) file defines a Kubeflow pipeline for automating the COPowereD XAI analysis workflow. This pipeline orchestrates the following components:

- Download Component: Downloads project files and data from a specified GitHub repository and branch. The pipeline expects the repo to contain:
  - Project files in root: `COPowereD_model.py`, `explainer.py`, `fairness_bias_analysis.py`, `fairness_bias_visualization.py`, `explainer_visualization.py`, `utils.py`.
  - Data in `data/` folder: `tabular_data.csv` (with labels), `pred.csv` (predictions).
- Fairness/Bias Analysis: Executes the fairness and bias analysis using the provided script, generating the output mentioned in the [Fairness and Bias Analysis Output](#fairness-and-bias-analysis-output) section.
- Fairness/Bias Visualization: Creates visual representations of fairness / bias metrics across demographic groups (Age, Gender), generating consolidated bar charts showing equalized odds (false positive rates) and demographic parity (prediction rates). Outputs PNG files described in [Fairness and Bias Analysis Visualizations](#fairness-and-bias-analysis-visualizations).
- Explainability Analysis: Executes the explainability analysis using the provided script, generating the output mentioned in the [Explainability Analysis Output](#explainability-analysis-output) section.
- Explainability Visualization: Generates visualizations based on the selected method (determined by sensitivity parameter):
  - SHAP (sensitivity ≥ 0.5): Beeswarm and bar plots showing feature impacts.
  - LIME (sensitivity < 0.5): Bar plot showing aggregated feature importance.
  - Outputs PNG files described in [Explainability Analysis Visualizations](#explainability-analysis-visualizations).

### Pipeline Architecture
The pipeline follows this execution pattern:
- Sequential Phase: Repository download runs first.
- Parallel Phase: Fairness and explainability analyses run simultaneously after repository download completes.
- Visualization Phase: Each analysis step is followed by its corresponding visualization step:
  - Fairness analysis followed by Fairness visualization
  - Explainability analysis followed by Explainability visualization

![Kubeflow Pipeline](./media/kubeflow_pipeline.png)

### Running the Pipeline
The pipeline can be compiled and deployed to a Kubeflow environment by executing:
```bash
python kubeflow_component/copowered_pipeline_component.py
```

The execution will generate a YAML file, `copowered_pipeline.yaml`, which can be used to create a new pipeline in Kubeflow by uploading the file through the Kubeflow UI.

The Kubeflow UI expects the following pipeline run parameters (arguments) when running:
- `github_repo_url`: URL of the GitHub repository containing the project files and data.
- `branch`: The branch of the GitHub repository to use. Default is `main`.
- `sensitivity`: Sensitivity value (between 0 and 1) to determine the explainability technique to be used. Default is `0.7`.

![Kubeflow Arguments](./media/kubeflow_parameters.png)


### Accessing the Generated Artifacts

The pipeline stores generated artifacts in MinIO object storage within the Kubeflow namespace. To access these artifacts:

- Set up port forwarding to the MinIO service by running `kubectl port-forward -n kubeflow svc/minio-service 9000:9000` in a terminal window
- Access the MinIO web interface at http://localhost:9000
- Log in with the default credentials: username: `minio`, password: `minio123`
- Navigate to the mlpipeline bucket, where you'll find the generated folders and files from each pipeline step, according to the automatically assigned uuid of the pipeline. (An example location could be: `http://localhost:9000/minio/mlpipeline/v2/artifacts/copowered-model-fairness-bias-and-explainer-pipeline/6d0fec69-aff6-487d-b717-71d45e25c71f/`)


## 📜 License & Usage

All rights reserved by MetaMinds Innovations.