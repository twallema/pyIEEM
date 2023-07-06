# Description of datasets

## Raw

### epi

#### shape

+ `SWE.json`: geojson file containing shapes of Swedish counties. Retrieved from: https://geodata.lib.utexas.edu/catalog/stanford-gp821bc1566

+ `conversion_SWE.py`: Script to clean the raw geojson shape file for Sweden.



#### demographic

+ `export.csv`: Demography of Belgium in 2019. Stratified per year of age from 0 to 100 years old. Stratified per province denoted by its NIS code. Retrieved from: https://bestat.statbel.fgov.be/bestat/ > Population by place of residence, nationality (Belgian/non-Belgian), marital status, age and gender 

+ `BE0101N1_20230704-171432.csv`: Demography of Sweden in 2019. Stratified per year of age from 0 to 100 years old. Stratified per county. Retrieved from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__BE__BE0101__BE0101C/

+ `BE0101U1_20230706-191203.csv`: Population, land area and population density per Swedish county in 2019. Retrieved from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__BE__BE0101__BE0101C/

#### labor_market_composition

+ `AM0207I5_20230705-155636.csv`: Gainfully employed 16+ years by Swedish county of residence (RAMS) and industrial classification NACE Rev. 2. Data for 2018. Retrieved from: https://www.statistikdatabasen.scb.se 

+ `AM0207I6_20230706-115023.csv`: Gainfully employed 16+ years by Swedish county of work (RAMS) and industrial classification NACE Rev. 2. Data for 2018. Retrieved from: https://www.statistikdatabasen.scb.se 

+ `sector_structure_by_work_BE_raw.csv`: C19: Werkende bevolking volgens plaats van tewerkstelling (provincies/buitenland), geslacht, economische sector (C), opleidingsniveau, land van staatsburgerschap (A) en leeftijd (B). Data likely resulting from the 2011 Census, however the origins of these data are not clear from Statbel's website. Retrieved from: https://bestat.statbel.fgov.be/bestat/crosstable.xhtml?datasource=f7fa1111-a328-454f-95f3-6c258f522754 

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

#### mobility

##### BE

+ `Pop_LPW_NL_25FEB15.XLSX`: contains the working population of Belgium per sex, place of residence and place of work. Data retrieved from: https://census2011.fgov.be/download/downloads_nl.html

+ `active_population_2011.csv`: contains the 18-60 yo (active) population of Belgium per province. Per one year age groups.

##### SWE

+ `AM0207AC_20230705-141955.csv`: Gainfully employed commuters in Sweden by county 16+ years by County of residence, County of work, sex and year. For 2018. Extracted from: https://www.statistikdatabasen.scb.se 

### eco

#### pichler

+ `table_ratio_inv_go.csv` contains, for every sector in the WIOD 55 classification, the number of days production can continue when no inputs are delivered (= stock). Retrieved from https://zenodo.figshare.com/articles/software/Production_networks_and_epidemic_spreading_How_to_restart_the_UK_economy_/12345527/1

+ `WIOD_shockdata.csv` contains estimated household and other demand shocks during an economic crisis. Retrieved from https://zenodo.figshare.com/articles/software/Production_networks_and_epidemic_spreading_How_to_restart_the_UK_economy_/12345527/1

+ `IHS_Markit_results_compact.csv` Criticality scores of IHS Markit analysts. The exact formulation of the question was as follows: “For each industry in WIOD 55, please rate whether each of its inputs are essential. We will present you with an industry X and ask you to rate each input Y. The key question is: Can production continue in industry X if input Y is not available for two months?” UK data, WIOD 55 classification. Retrieved from https://zenodo.figshare.com/articles/software/Production_networks_and_epidemic_spreading_How_to_restart_the_UK_economy_/12345527/1

#### national_accounts

##### BE

+ `employment_NACE64_2019.csv`: Number of employees per economic activity of NACE64 in Belgium (2019). Retrieved from NBB.stat > Population and Labour Market > Employment > Employment: annual detailed data > Domestic concept: A64.

+ `vR64_nl_20191213.xlsx`: Input-ouput tables for Belgium. NACE 64 classification. 2015. Retrieved from https://www.plan.be/databases/data-63-en-input_output_tables_2015

##### SWE

`nrio_siot_181108.xlsx` Symmetric Input-ouput tables for Sweden. NACE 64 classification. 2016. Retrieved from: https://www.scb.se/en/finding-statistics/statistics-by-subject-area/national-accounts/national-accounts/national-accounts-previous-definitions/pong/tables-and-graphs/input-output-tables-2008-2016/ > Symmetrical tables - SIOT, 2008-2016 (xlsx)

#### calibration_data

##### BE

##### SWE

+ `AM04011G_20230706-132742.csv`: Unemployed persons aged 15-74 (LFS) in Sweden by month. Retrieved from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__AM__AM0401__AM0401L/

+ `AM0401UL_20230706-133417.csv`: Unemployed (number of persons), age stratified. Month 2001M01 - 2023M05. Retrieved from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__AM__AM0401__AM0401O/NAKUAkrUtbudM/ 

+ `AM0401UL_20230706-133646.csv`: Not fully employed (number of persons), age stratified. Month 2001M01 - 2023M05. Retrieved from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__AM__AM0401__AM0401O/NAKUAkrUtbudM/ 


## Interim

### epi

#### shape

+ `SWE.json`: Cleaned shape file for Sweden, generated from `~/data/raw/epi/shape/SWE.json` using `~/data/raw/epi/shape/conversion_SWE.py`. Contains the population density per county.

#### demographic

+ `age_structure_BE_2019.csv`: Demography of Belgium in 2019. Stratified per year of age from 0 to 100 years old. Stratified per province denoted by its name. Cleaned version of raw demography. 

+ `age_structure_SWE_2019.csv`: Demography of Sweden in 2019. Stratified per year of age from 0 to 100 years old. Stratified per county denoted by its name. Cleaned version of raw demography. 

+ `population_density_SWE_2019.csv`: Population, land area and population density per Swedish county in 2019. Cleaned version of `BE0101U1_20230706-191203.csv`.

##### proximity

`verbeeck_incidences_NACE64.xlsx`: Contains the extracted 14-Day incidence of COVID-19 in sectors with minimum 10,000 employees at NACE Level 1/2 prior to the 19th October non-pharmaceutical interventions (Table S1/S2 + Figure 1), formatted using the raw data in `~data/raw/epi/contacts/helicon/incidences.xlsx` to match the NACE 64 economic activities. Normalised with the incidence over all sectors. The following assumptions were made during the conversion: B05-09, equal to C; C10-12, equal to average of C1011, C1012, C1013; N80-82: equal to average N80 and N82; Q87-88: equal to average of Q and Q87; R90-92: equal to average of R and R90; T97-98: equal to S96. Tab `20200929_20201012` contains incidences collected between Sep. 29, 2020 and Oct. 12, 2020, tab `20201006_20201019` contains incidences collected between Oct. 6, 2020 and Oct. 19, 2020. Tab `comparison` normalises the relative incidences so that the employee-weighted average relative incidence sums to one.

+ `pichler_figure_S5_NACE64.csv`: Physical Proximity Index (PPI) of social contacts converted to the NACE64 classification. Assumptions made during conversion: 1) N77, N78, N79 equal to N. 2) T97-98 equal to N. R90-92, R93 equal to I. S94, S95 equal to N. S96 equal to the average of N and G47. Employee-weighted average physical proximity of R90-92, R93, S94, S95, S96 (R_S) is 0.655 under the above assumptions, which is almost identical to the physical proximity used by Pichler of 0.645.

+ `pichler_table_5_NACE64.csv`: Remote Labor Index (RLI) of activities converted to the NACE64 classification. Assumptions made during conversion: 1) N77, N78, N79 equal to N. 2) S94, S95, S96 equal to G47. 3) R90-92, R93: Set to 0.05 to obtain an employee-weighted average remote labor index of R90-92, R93, S94, S95, S96 (R_S) of 39, which is equal to the value used by Pichler. 4) T: Set to 0.05. Essentiality score of activities converted to the NACE64 classification. Assumptions made during conversion: 1) N77, N78, N79 equal to N. 2) R90-92, R93 equal to I. 3) S94, S95, S96: Set to 63 to obtain an employee-weighted average essentiality score for R90-92, R93, S94, S95, S96 (R_S) of 47, which is equal to the value used by Pichler.

`estimated_remote_labor.csv`: Estimated fraction of workers able to work from home. Primarily based on the reported fraction of workers working from home during the first COVID-19 lockdown in Belgium (`ermg-tables.xlsx`), when telework was mandated where possible. The Remote Labor Index from `pichler_table_5_NACE64.csv` was used to infer the fraction of telework attainable when data was available. Assumptions listed in the data file.

#### labor_market_composition

+ `sector_structure_by_residence_SWE_2019.csv`: Number (and fraction of the county's population) of gainfully employed by Swedish county of residence (RAMS) and industrial classification NACE Rev. 2. Data for 2018. Cleaned version of `AM0207I5_20230705-155636.csv`. I verified the relative fractions in every county sum to one.
sector_structure_by_work_SWE_2019.csv

+ `sector_structure_by_work_SWE_2019.csv`: Number (and fraction of the county's total number of workers) of gainfully employed by Swedish county of work (RAMS) and industrial classification NACE Rev. 2. Data for 2018. Cleaned version of `AM0207I6_20230706-115023.csv`. I verified the relative fractions in every county sum to one.

+ `sector_structure_by_work_BE.csv`: Cleaned version of `sector_structure_by_work_BE_raw.csv`.

#### mobility

##### BE

+ `active_population_2011_format.csv`: contains the 18-60 yo (active) population of Belgium per province. Summed from 18-60 years old.

+ `Pop_LPW_NL_25FEB15_delete_unknown.xlsx`: from the raw spreadsheet `Pop_LPW_NL_25FEB15.xlsx`, the columns denoting the inhabitants with an unkown location of work were removed.

+ `extract_recurrent_mobility.py`: script to extract and normalise the recurrent mobility matrix for Belgium.

+ `recurrent_mobility_BE.csv`: contains the recurrent mobility matrix between the Belgian provinces, obtained from the census of 2011. Note that the rows do not sum to one as not every inhabitant has a job. This effect is not minor, with as low as 52% (!) of the active population (18-60 yo) not having a job. The national average employment rate in Sweden is 68.7%.

##### SWE

+ `AM0207AC_20230705-141955_format.csv`:  Gainfully employed commuters in 2018 by Swedish county 16+ years by County of residence, County of work, sex and year. For 2018. Formatted in an origin-destination style matrix. Raw data extracted from: https://www.statistikdatabasen.scb.se 

+ `active_population_2019_format.csv`: Number of Swedish inhabitants between 16-60 year old (active population) in 2019. Used to normalize the recurrent mobility matrix. Formatted from `age_structure_SWE_2019.csv`. 

+ `recurrent_mobility_SWE.csv`: contains the normalized recurrent mobility matrix between the Swedish counties for 2018. Note that the rows do not sum to one as not every inhabitant has a job. This effect is not minor, with as low as 52% (!) of the active population (18-60 yo) not having a job. The national average employment rate in Sweden is 84.5%.

### eco

#### national_accounts

`conversion_matrix.csv`: Convert from the Swedish version of NACE64 (only 57 out of 63 sectors available) to the Belgian version. The Swedish data sadly lumps some of the NACE 64's economic activities: C20, C21 --> C20-21; G45, G46, G47 --> G45-47; H52, H53 --> H52-53; M71, M72 --> M71-72. Belgian simulations must be aggregated before comparing them to Sweden.

##### BE

+ `IO_BE_NACE64.csv`: symmetric input-output table for Belgium, formatted to NACE 64 classification starting from `vR64_nl_20191213.xlsx`. Values from tab Tbl_8 "Symmetrische input-outputtabel (product x product)". Sector L68 is split in two in the raw IO table: '68_' (real estate minus rent) and '68a' (rent). In the formatted IO table, rent is ommitted for the following reason: Under a pandemic shock households will not stop paying rent, either because they retain their income, or because the govnerment furloughs them (in either case people didn't get thrown out of their houses during the COVID-19 crisis). Ommitting rent from L68 implies that L68 in our model represents the buying-a-house on-site consumption type of real estate activities.

`other_accounts_BE_NACE64.csv`: All other variables from the Belgian national accounts `vR64_nl_20191213.xlsx` needed to initialize the model. 

##### SWE

`IO_SWE_NACE64.csv`: symmetric input-output table for Sweden, formatted to NACE 64 classification starting from `nrio_siot_181108.xlsx`. The Swedish data sadly lumps some of the NACE 64's economic activities: C20, C21 --> C20-21; G45, G46, G47 --> G45-47; H52, H53 --> H52-53; M71, M72 --> M71-72. Similarily to to the Belgian IO matrix, revenue from rent was removed from L68.

`other_accounts_SWE.csv`: All other variables from the Swedish national accounts `nrio_siot_181108.xlsx` needed to initialize the model. 

#### pichler

+ `IHS_critical_NACE64.csv` contains the IHS Market Analysts data (`IHS_Markit_results_compact.csv`), reformatted from WIOD 55 to the NACE 64 classification. Columns represent the critical inputs to a sector. Dependecy of L68 (real estate) on H53 (Postal services) was removed. The Real estate sector did not face a big decline in economic activity, as detailed in https://www.nbb.be/doc/ts/publications/economicreview/2021/ecorevii2021.pdf (chart 5). Dependecy of H49 (Land Transport) and H51 (Air Transport) on I55-56 (Accodomodation) was removed as the closure of Accomodation during the lockdown led to overly large declines in economic activity. Sector Rental and Leasing (N77) critically depends on C33 (Repair of Machinery) and G45 (Retail of vehicles) only. Travel agencies (N79) critically depend on Air, Water and Land transport (H49/H50/H51) and importantly depend on I55-56, N77/N78. All dependencies of Public Administration (O84) and Education (P85) were relaxed as the government sector didn't face any shocks during the COVID-19 pandemic (https://www.nbb.be/doc/ts/publications/economicreview/2021/ecorevii2021.pdf).

+ `desired_stock_NACE64.csv`: For every sector in the NACE 64 classification, the number of days production can continue when no inputs are delivered (= stock). Converted version of `table_ratio_inv_go.csv`.

+ `on_site_consumption_NACE64.csv`: For every sector in the NACE 64 classification, if the consumption happens on-site.

#### misc

+ `conversion_matrices.xlsx` contains conversion matrices to aggregate data from different economic activity classifications. F.i. converting from NACE 64 to WIOD 55 classification. Only works for Belgium.
