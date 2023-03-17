**Group 20:** Amitash Nanda, Dhruv Talwar, Haonan Li, Zhenzhe He 
# Circadian-Rhythm-driven-Alertness-Investigation
*A Study based on how alertness is affected by different factors in life, including sleep, food and exercise.*

*Dataset: [Alertness data](https://github.com/amitashnanda/Circadian-Rhythm-driven-Alertness-Investigation/blob/main/data/data.csv)*


## **File Structure**

```
.
├── README.md               <- README
│
├── Data
│   ├──Raw_Data.xlsx
│   ├──data.csv
│   ├──data143_final.xlsx
│   ├──ece_143_data.xlsx
│   └──predicted_result.csv
│
├── src                   <- src files to create the plots and train models
│   ├──Visualization.ipynb
│   ├──demographics_sleep.ipynb
│   ├──descriptive_stats.ipynb
│   ├──exercise_food.ipynb
│   ├──heritability.ipynb
│   ├──model.py
│   ├──model_final.ipynb
│   └──new_model.py 
│
├── plots         <- Plots created by files in src
│
├── res           <- Prediction results and all plots from visualization
│
└── ECE 143 Project Presentation.pdf        <- PDF of presentation
```

---
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
conda env create -f requirement.yaml
```

## How to run
 
1. Change the path of raw_data and save_path if needed
2. Install environment(Please check Prerequisites section)
3. Run model.py
4. The result csv file will be saved to your save_path
5. If wish to check the individual factor analysis, open jupytor notebook to and run individual files. 
```
jupyter nbconvert --execute <notebook>
```
6. If wish to check all the plots, open jupyter notebook and run Visualization.ipynb
```
jupyter nbconvert --execute Visualization.ipynb
```



## **Sources**

*[D. Raca, D. Leahy, C.J. Sreenan and J.J. Quinlan. Beyond Throughput, The Next Generation: A 5G Dataset with Channel and Context Metrics. ACM Multimedia Systems Conference (MMSys), Istanbul, Turkey. June 8-11, 2020.](https://doi.org/10.1038/s41467-022-34503-2)*
