# Description of datasets

## Raw

### epi

#### demographic

##### BE

+ `export.csv`: Demography of Belgium in 2019. Stratified per year of age from 0 to 100 years old. Stratified per province denoted by its NIS code. Retrieved from: https://bestat.statbel.fgov.be/bestat/ > Population by place of residence, nationality (Belgian/non-Belgian), marital status, age and gender 

+ `BE0101N1_20230704-171432.csv`: Demography of Belgium in 2019. Stratified per year of age from 0 to 100 years old. Stratified per county. Retrieved from:

##### SWE


#### contacts

##### proximity

+ `verbeeck_figure_1.jpg`: The 14-day incidence of COVID-19 in selected sectors in periods 29 September–12 October and 6–19 October. Figure 1 in "Verbeeck J, Vandersmissen G, Peeters J, Klamer S, Hancart S, Lernout T, Dewatripont M, Godderis L, Molenberghs G. Confirmed COVID-19 Cases per Economic Activity during Autumn Wave in Belgium. Int J Environ Res Public Health. 2021 Nov 27;18(23):12489."

+ `ijerph-1447404-supplementary.pdf`: Contains the 14-Day incidence of COVID-19 in sectors with minimum 10,000 employees at NACE Level 1/2 prior to the 19th October non-pharmaceutical interventions (Table S1/S2). Can be used to estimate the relative risk of infection in every stector of the economy. Retrieved from the supplementary materials of "Verbeeck J, Vandersmissen G, Peeters J, Klamer S, Hancart S, Lernout T, Dewatripont M, Godderis L, Molenberghs G. Confirmed COVID-19 Cases per Economic Activity during Autumn Wave in Belgium. Int J Environ Res Public Health. 2021 Nov 27;18(23):12489."

+ `verbeeck_incidences.xlsx`: Contains the extracted 14-Day incidence of COVID-19 in sectors with minimum 10,000 employees at NACE Level 1/2 prior to the 19th October non-pharmaceutical interventions (Table S1/S2 + Figure 1). Tab `20200929_20201012` contains incidences collected between Sep. 29, 2020 and Oct. 12, 2020, tab `20201006_20201019` contains incidences collected between Oct. 6, 2020 and Oct. 19, 2020.

+ `pichler_figure_S5.png`: Physical Proximity Index of social contacts in WIOD55 industries. Obtained from the Supplementary Materials of "Anton Pichler, Marco Pangallo, R. Maria del Rio-Chanona, François Lafond, J. Doyne Farmer. (2022). Forecasting the propagation of pandemic shocks with a dynamic input-output model. Journal of Economic Dynamics and Control, 144".

+ `pichler_figure_S5.csv`: Physical Proximity Index of social contacts in WIOD55 industries, extracted from `figure_S5.png`.

+ `pichler_Table_5.png`: Remote Labor Index (RLI) and Essentiality Score of activities in WIOD55 industries. Obtained from "Anton Pichler, Marco Pangallo, R. Maria del Rio-Chanona, François Lafond, J. Doyne Farmer. (2022). Forecasting the propagation of pandemic shocks with a dynamic input-output model. Journal of Economic Dynamics and Control, 144".

+ `pichler_table_5.png`: Remote Labor Index (RLI) and Essentiality Score of activities in WIOD55 industries, exracted from `pichler_Table_5.png`.

+ `ermg-tables.xlsx` Ecomical Risk Management Group (ERMG) business survey indicators. Series of surveys conducted by the ERMG to assess the economic impact of COVID-19. Retrieved from https://www.nbb.be/nl/ermg-enquetes 

##### matrices

Underlying folder structure ommitted from this readme. Contains the interaction matrix (in the place and time suggested by the folder and spreadsheets names) based on the 2008 study by Joel Mossong for Belgium and Finland. The spreadsheet has several tabs to distinguish between the nature and duration of the contact. Data extracted using the social contact rates data tool made by Lander Willem, available at https://lwillem.shinyapps.io/socrates_rshiny/. During extraction of the data, weighing by age, weighing by week/weekend were used and reciprocity was assumed. Both physical and non-physical contacts were included.

### eco

+ `employment_NACE64_2019.csv`: Number of employees per economic activity of NACE64 in Belgium (2019). Retrieved from NBB.stat > Population and Labour Market > Employment > Employment: annual detailed data > Domestic concept: A64.

## Interim

### epi

#### contacts

##### proximity

`verbeeck_incidences_NACE64.xlsx`: Contains the extracted 14-Day incidence of COVID-19 in sectors with minimum 10,000 employees at NACE Level 1/2 prior to the 19th October non-pharmaceutical interventions (Table S1/S2 + Figure 1), formatted using the raw data in `~data/raw/epi/contacts/helicon/incidences.xlsx` to match the NACE 64 economic activities. Normalised with the incidence over all sectors. The following assumptions were made during the conversion: B05-09, equal to C; C10-12, equal to average of C1011, C1012, C1013; N80-82: equal to average N80 and N82; Q87-88: equal to average of Q and Q87; R90-92: equal to average of R and R90; T97-98: equal to S96. Tab `20200929_20201012` contains incidences collected between Sep. 29, 2020 and Oct. 12, 2020, tab `20201006_20201019` contains incidences collected between Oct. 6, 2020 and Oct. 19, 2020. Tab `comparison` normalises the relative incidences so that the employee-weighted average relative incidence sums to one.

+ `pichler_figure_S5_NACE64.csv`: Physical Proximity Index (PPI) of social contacts converted to the NACE64 classification. Assumptions made during conversion: 1) N77, N78, N79 equal to N. 2) T97-98 equal to N. R90-92, R93 equal to I. S94, S95 equal to N. S96 equal to the average of N and G47. Employee-weighted average physical proximity of R90-92, R93, S94, S95, S96 (R_S) is 0.655 under the above assumptions, which is almost identical to the physical proximity used by Pichler of 0.645.

+ `pichler_table_5_NACE64.csv`: Remote Labor Index (RLI) of activities converted to the NACE64 classification. Assumptions made during conversion: 1) N77, N78, N79 equal to N. 2) S94, S95, S96 equal to G47. 3) R90-92, R93: Set to 0.05 to obtain an employee-weighted average remote labor index of R90-92, R93, S94, S95, S96 (R_S) of 39, which is equal to the value used by Pichler. 4) T: Set to 0.05. Essentiality score of activities converted to the NACE64 classification. Assumptions made during conversion: 1) N77, N78, N79 equal to N. 2) R90-92, R93 equal to I. 3) S94, S95, S96: Set to 63 to obtain an employee-weighted average essentiality score for R90-92, R93, S94, S95, S96 (R_S) of 47, which is equal to the value used by Pichler.

`estimated_remote_labor.csv`: Estimated fraction of workers able to work from home. Primarily based on the reported fraction of workers working from home during the first COVID-19 lockdown in Belgium (`ermg-tables.xlsx`), when telework was mandated where possible. The Remote Labor Index from `pichler_table_5_NACE64.csv` was used to infer the fraction of telework attainable when data was available. Assumptions listed in the data file.

### eco

