# Description of datasets

## Raw

### epi

#### cases

+ `statistik-covid19-inskrivna.xlsx` Number of admissions to the hospital per week in the Swedish counties. Retrieved from: https://www.socialstyrelsen.se/globalassets/sharepoint-dokument/dokument-webb/statistik/statistik-covid19-avlidna.xlsx 

+ `41598_2021_3499_MOESM1_ESM.pdf`: Figure S2 contains the weekly number of COVID-19 hospital admission in the 20 regions of Sweden. Supplementary materials of: "Predicting regional COVID-19 hospital admissions in Sweden using mobility data."

+ `COVID19BE_HOSP.csv`:  Hospitalisations for COVID-19 in Belgium by date and provinces. Retrieved from: https://epistat.sciensano.be/covid/

#### shape

##### SWE

+ `SWE.json`: geojson file containing shapes of Swedish counties. Retrieved from: https://geodata.lib.utexas.edu/catalog/stanford-gp821bc1566

+ `conversion_SWE.py`: Script to clean the raw geojson shape file for Sweden.

##### BE

+ Shape files containing shapes of Belgian provinces. Retrieved from http://www.diva-gis.org/datadown

+ `conversion_BE.py`: Script to clean the raw shape file for Belgium. Limburg province was split in two for some weird reason so this had to be resolved.

#### demographic

+ `export.csv`: Demography of Belgium in 2019. Stratified per year of age from 0 to 100 years old. Stratified per province denoted by its NIS code. Retrieved from: https://bestat.statbel.fgov.be/bestat/ > Population by place of residence, nationality (Belgian/non-Belgian), marital status, age and gender 

+ `BE0101N1_20230719-185419.csv`: Demography of Sweden in 2019. Total population per county. Retrieved from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__BE__BE0101__BE0101A 

+ `BE0101N1_20230704-171432.csv`: Demography of Sweden in 2019. Stratified per year of age from 0 to 100 years old. Stratified per county. Retrieved from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__BE__BE0101__BE0101C/

+ `BE0101U1_20230706-191203.csv`: Population, land area and population density per Swedish county in 2019. Retrieved from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__BE__BE0101__BE0101C/

+ `pop-totale-france-metro.xlsx`: demography of metropolitan france per year of age and sex in 2019. Used to distribute the Supplementary Professional Contacts across the desired age groups of the contact matrices. Retrieved from: https://www.insee.fr/en/statistiques/2382599?sommaire=2382613. Modified: saved as .xlsx instead of .xls.

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

##### BE & FIN

Contains the interaction matrix (in the place and time suggested by the folder and spreadsheets names) based on the 2008 study by Joel Mossong for Belgium and Finland. The spreadsheet has several tabs to distinguish between the nature and duration of the contact. Data extracted using the social contact rates data tool made by Lander Willem, available at https://lwillem.shinyapps.io/socrates_rshiny/. During extraction of the data, weighing by age, weighing by week/weekend were used and reciprocity was assumed. Both physical and non-physical contacts were included.

##### FR

+ `RawData_ComesF.xlsx`: contains the raw dataset from "The French Connection: The First Large Population-Based Contact Survey in France Relevant for the Spread of Infectious Diseases". Social contact data is available based on the place of work. Modifications: 1) Merge AS --> AY, rename 'SPC'. 2) Rename 'contacts  x' (double space is intentional!), going from 1 to 40 and then again from 1 to 29 --> 'contacts 1' --> 'contacts 69'. 

+ `S1_Fig.pdf`: questionnaire used by the authors to obtain the social contact data in `RawData_ComesF.xlsx`.

+ `format_raw_data.py`: Converts the raw data in `RawData_ComesF.xlsx` in a more Python-friendly format for further processing. The analysis is run using 5-year age intervals. Noteable asumptions: 1) Maximum SPC contacts truncated at 200, opposed to the original author who set this to 134 (95% quantile of reported SPC contacts). I opted for 200 because cutting of the top 5% observations from a population whos distribution will have a long tail frankly seems statistically wrong to me. 2) Duration distribution of SPC contacts assumed the same as the distribution for work_indoor contacts of survey particpants reporting between 10 and 20 contacts. Outputs a large file `comesf_formatted_data.csv`, which is subsequently used as input by `regress_formatted_data.py`.

+ `regress_formatted_data.py`: Peforms a Negative Binomial Regression to the data using a Generalised Estimating Equation. The regression is performed in every location seperately because of limits in computational resources. The data are then expanded to contain a value for every age x and age y. Outputs a large file `comesf_raw_matrices.csv`, which is subsequently used as input by `postprocess_raw_matrices.py`.

+ `postprocess_raw_matrices.py`: (1) The locations 'transport' and 'work_leisure_outdoor' are eliminated by merging into other locations. Transport contacts for children during a regular week are merged with the school contacts. Vacation and weekend contacts of children are merged with 'leisure_public'. Transport contacts for elderly are always merged with 'leisure_public'. Transport contacts for the working population during the week (irregardless of vacation) are merged with 'work_indoor' while weekend contacts are merged with 'leisure_public'. Outdoor work/leisure contacts for the active population are merged with 'work_indoor' during the week and with 'leisure_private' during the weekend. Contacts for youths and elderly are merged with 'leisure_private'. (2) The 'SPC' contacts, for which we have assumed that they happen during weekdays outside vacations, are distributed across weekends/vacation according to the observed ratio weekday/weekendday vacation/no vacation on contacts 'work indoor'. (3) The durations are eliminated, the absolute number of contacts and integrated number of contact minutes are saved. (4) The week/weekend average is added to the dataset. (5) The sample size for sector 'K' was small, therefore the data were averaged with the highly similar sector 'M'. (6) Reciprocity at home, leisure_private and leisure_public is forced. (7) The observed sectors are expanded to every level of NACE 21. The results are saved as `data/interim/epi/contacts/matrices/FR/comesf_formatted_matrices.csv`.

#### mobility

##### BE

+ `Pop_LPW_NL_25FEB15.XLSX`: contains the active population of Belgium per sex, place of residence and place of work. Data retrieved from: https://census2011.fgov.be/download/downloads_nl.html

##### SWE

+ `AM0207AC_20230705-141955.csv`: Gainfully employed commuters in Sweden by county 16+ years by County of residence, County of work, sex and year. For 2018. Extracted from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__AM__AM0207__AM0207L/AM0207PendlLanA04/

### eco

#### other_demand

+ `COMEXT_17082023124307665.csv`: Monthly foreign trade data for Belgium. Use to inform the magnitude of the export shock during the 2020 COVID-19 pandemic. Retrieved from: https://stat.nbb.be/Index.aspx?DataSetCode=EXTTRADEBENAT#

+ `DP_LIVE_16082023121712365.csv`: Quarterly investment data by OECD (Quarter-over-Quarter) for BE, SWE and the EU27. Retrieved from: https://data.oecd.org/gdp/investment-gfcf.htm#indicator-chart 

+ `figure_2_exports_OECD.png`: Change in exports of goods and services during the COVID-19 pandemic, normalised with the export in Jan 2020. (G7 countries). Retrieved from: https://www.oecd.org/coronavirus/policy-responses/international-trade-during-the-covid-19-pandemic-big-shifts-and-uncertainty-d1131663/#figure-d1e95

+ `ha0201_2023m06_di_01_en.xls`: Monthly foreign trade data for Sweden (Trend -> Export). Use to inform the magnitude of the export shock during the 2020 COVID-19 pandemic.

#### labor_market_composition

+ `AM0207I5_20230705-155636.csv`: Gainfully employed 16+ years by Swedish county of residence (RAMS) and industrial classification NACE Rev. 2. Data for 2018. Retrieved from: https://www.statistikdatabasen.scb.se 

+ `AM0207I6_20230706-115023.csv`: Gainfully employed 16+ years by Swedish county of work (RAMS) and industrial classification NACE Rev. 2. Data for 2018. Retrieved from: https://www.statistikdatabasen.scb.se 

+ `sector_structure_by_work_BE_raw.csv`: C19: Werkende bevolking volgens plaats van tewerkstelling (provincies/buitenland), geslacht, economische sector (C), opleidingsniveau, land van staatsburgerschap (A) en leeftijd (B). Data likely resulting from the 2011 Census, however the origins of these data are not clear from Statbel's website. Retrieved from: https://bestat.statbel.fgov.be/bestat/crosstable.xhtml?datasource=f7fa1111-a328-454f-95f3-6c258f522754 

`active_population_BE_raw.csv`: Employed fraction 15-64 year olds. Fraction of total population between 15-64 jaar. Total population. Per Belgian province. Census of 2011. Retrieved from: https://bestat.statbel.fgov.be/bestat/crosstable.xhtml?datasource=06deb4bd-8f91-49fb-befb-cfb25108b5ae (dataset: "Geografische indicatoren (gebaseerd op Census 2011)")

#### pichler

+ `table_ratio_inv_go.csv` contains, for every sector in the WIOD 55 classification, the number of days production can continue when no inputs are delivered (= stock). Retrieved from https://zenodo.figshare.com/articles/software/Production_networks_and_epidemic_spreading_How_to_restart_the_UK_economy_/12345527/1

+ `WIOD_shockdata.csv` contains estimated household and other demand shocks during an economic crisis. Retrieved from https://zenodo.figshare.com/articles/software/Production_networks_and_epidemic_spreading_How_to_restart_the_UK_economy_/12345527/1

+ `IHS_Markit_results_compact.csv` Criticality scores of IHS Markit analysts. The exact formulation of the question was as follows: “For each industry in WIOD 55, please rate whether each of its inputs are essential. We will present you with an industry X and ask you to rate each input Y. The key question is: Can production continue in industry X if input Y is not available for two months?” UK data, WIOD 55 classification. Retrieved from https://zenodo.figshare.com/articles/software/Production_networks_and_epidemic_spreading_How_to_restart_the_UK_economy_/12345527/1

#### national_accounts

##### BE

+ `employment_NACE64_2019.csv`: Number of employees per economic activity of NACE64 in Belgium (2019). Retrieved from NBB.stat > Population and Labour Market > Employment > Employment: annual detailed data > Domestic concept: A64.

+ `vR64_nl_20191213.xlsx`: Input-ouput tables for Belgium. NACE 64 classification. 2015. Retrieved from https://www.plan.be/databases/data-63-en-input_output_tables_2015

##### SWE

`nrio_siot_181108.xlsx` Symmetric Input-ouput tables for Sweden. NACE 64 classification but some sectors are aggregated so there are only 57 sectors in total :'( . 2016. Contains the total number of employees per sector. Retrieved from: https://www.scb.se/en/finding-statistics/statistics-by-subject-area/national-accounts/national-accounts/national-accounts-previous-definitions/pong/tables-and-graphs/input-output-tables-2008-2016/ > Symmetrical tables - SIOT, 2008-2016 (xlsx)

`input_output_produkt_2015-2020_korr.xlsx` Symmetric Input-ouput tables for Sweden, 2015-2020. NACE 64 classification. Retrieved from: https://www.scb.se/en/finding-statistics/statistics-by-subject-area/national-accounts/national-accounts/national-accounts-quarterly-and-annual-estimates/pong/tables-and-graphs/tables/input-output-tables-productproduct/#:~:text=Tabellerna%20inneh%C3%A5ller%20symmetriska%20input%2Doutput,2%20och%20CPA%202008.

#### calibration_data

##### BE

##### SWE

+ `am0401_2023m06_di_01_en_aku.xls`: Seasonally adjusted and smoothed monthly number of hours worked for Sweden. Retrieved from: https://www.scb.se/en/finding-statistics/statistics-by-subject-area/labour-market/labour-force-surveys/labour-force-surveys-lfs/pong/tables-and-graphs/seasonally-adjusted-data/time-series-on-the-number-of-hours-worked-persons-15-74-years/

+ `000000X3_20230817-103925.csv`: Seasonally adjusted monthly GDP data at constant prices (2011=100) for Sweden. Retrieved from: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__NR__NR9999__NR9999A/NR9999ENS2010BNPIndN/table/tableViewLayout1/

## Interim

### epi

#### cases

+ `hospital_incidence_BE.csv`: A copy of `COVID19BE_HOSP.csv` without modifications, renamed to the format `f'hospital_incidence_{country}.csv'` and placed along its Swedish equivalent to use less code.

+ `hospital_incidence_SWE.csv`: Number of hospital admissions per Swedish county and per week, converted from file `statistik-covid19-inskrivna.xlsx` by replacing all 'X' with zero.

#### shape

+ `SWE.json`: Cleaned shape file for Sweden, generated from `~/data/raw/epi/shape/SWE/SWE.json` using `~/data/raw/epi/shape/SWE/conversion_SWE.py`. Contains the population density per county.

+ `BE.json`: Cleaned shape file for Belgium, generated using `~/data/raw/epi/shape/BE/conversion_BE.py`. Contains the population density per province.

#### demographic

+ `age_structure_BE_2019.csv`: Demography of Belgium in 2019. Stratified per year of age from 0 to 100 years old. Stratified per province denoted by its name. Cleaned version of raw demography. 

+ `age_structure_SWE_2019.csv`: Demography of Sweden in 2019. Stratified per year of age from 0 to 100 years old. Stratified per county denoted by its name. Cleaned version of raw demography. 

+ `population_density_SWE_2019.csv`: Population, land area and population density per Swedish county in 2019. Cleaned version of `BE0101U1_20230706-191203.csv`.

+ `age_structure_FR_2019.csv`: Population of metropolitan France in 2019. Clean version of `pop-totale-france-metro.xlsx`. Elongated until 120 years of age.

#### contacts

##### matrices

###### FR

`comesf_formatted_matrices.csv`: Contains the contact matrices from the ComesF study. Stratified into: 1) five year age intervals, 2) locations: home, work, leisure_public, leisure_private, school. 3) Economic activity (NACE 21) - matrices at home, schools, leisure_public, leisure_private do not differ with economic activity. They are only present for every sector to make subsequent extraction more elegant. 4) Daytype: weekday, weekendday, average. 5) Vacation: True or False. 6) Absolute number of contacts or integrated contact duration.

##### proximity

`verbeeck_incidences_NACE64.xlsx`: Contains the extracted 14-Day incidence of COVID-19 in sectors with minimum 10,000 employees at NACE Level 1/2 prior to the 19th October non-pharmaceutical interventions (Table S1/S2 + Figure 1), formatted using the raw data in `~data/raw/epi/contacts/helicon/incidences.xlsx` to match the NACE 64 economic activities. Normalised with the incidence over all sectors. The following assumptions were made during the conversion: B05-09, equal to C; C10-12, equal to average of C1011, C1012, C1013; N80-82: equal to average N80 and N82; Q87-88: equal to average of Q and Q87; R90-92: equal to average of R and R90; T97-98: equal to S96. Tab `20200929_20201012` contains incidences collected between Sep. 29, 2020 and Oct. 12, 2020, tab `20201006_20201019` contains incidences collected between Oct. 6, 2020 and Oct. 19, 2020. Tab `comparison` normalises the relative incidences so that the employee-weighted average relative incidence sums to one.

+ `pichler_figure_S5_NACE64.csv`: Physical Proximity Index (PPI) of social contacts converted to the NACE64 classification. Assumptions made during conversion: 1) N77, N78, N79 equal to N. 2) T97-98 equal to N. R90-92, R93 equal to I. S94, S95 equal to N. S96 equal to the average of N and G47. Employee-weighted average physical proximity of R90-92, R93, S94, S95, S96 (R_S) is 0.655 under the above assumptions, which is almost identical to the physical proximity used by Pichler of 0.645.

+ `pichler_table_5_NACE64.csv`: Remote Labor Index (RLI) of activities converted to the NACE64 classification. Assumptions made during conversion: 1) N77, N78, N79 equal to N. 2) S94, S95, S96 equal to G47. 3) R90-92, R93: Set to 0.05 to obtain an employee-weighted average remote labor index of R90-92, R93, S94, S95, S96 (R_S) of 39, which is equal to the value used by Pichler. 4) T: Set to 0.05. Essentiality score of activities converted to the NACE64 classification. Assumptions made during conversion: 1) N77, N78, N79 equal to N. 2) R90-92, R93 equal to I. 3) S94, S95, S96: Set to 63 to obtain an employee-weighted average essentiality score for R90-92, R93, S94, S95, S96 (R_S) of 47, which is equal to the value used by Pichler.

+ `ermg_remote_labor.csv`: Fraction of workers working from home during the first Belgian 2020 COVID-19 lockdown (`ermg-tables.xlsx`). Where data was available, the average reported fraction of telework from four surveys performed in April 2020 were used. The Remote Labor Index from `pichler_table_5_NACE64.csv` was used to infer the fraction of telework attainable when no data was available. Assumptions listed in the data file.

+ `ermg_at_workplace.csv`: Fraction of workers working in the workplace during the first Belgian 2020 COVID-19 lockdown (`ermg-tables.xlsx`). Where data was available, the average reported fraction of telework from four surveys performed in April 2020 were used.Assumptions listed in the data file.

+ `ermg_summary.csv`: Combines `ermg_remote_labor.csv` and `ermg_at_workplace.csv` into a cleaner format.

+ `leisure_association_vectors.csv`: Contains the degree association between economic activities of leisurely nature and changes in contacts and consumption patterns.

#### mobility

##### BE

+ `Pop_LPW_NL_25FEB15_delete_unknown.xlsx`: from the raw spreadsheet `Pop_LPW_NL_25FEB15.xlsx`, the columns denoting the inhabitants with an unkown location of work were removed.

+ `extract_recurrent_mobility.py`: script to extract and normalise the recurrent mobility matrix for Belgium.

+ `recurrent_mobility_normtotal_BE.csv`: contains the recurrent mobility matrix between the Belgian provinces, obtained from the census of 2011. Normalised by the total population.

+ `recurrent_mobility_normactive_BE.csv`: contains the recurrent mobility matrix between the Belgian provinces, obtained from the census of 2011. Normalised by the total active population (16-65 yo).

##### SWE

+ `AM0207AC_20230705-141955_format.csv`:  Number of gainfully employed commuters in 2018 by Swedish county 16+ years by County of residence, County of work, sex and year. For 2018. Formatted from `AM0207AC_20230705-141955.csv` into an origin-destination style matrix. Raw data extracted from: https://www.statistikdatabasen.scb.se 

+ `BE0101N1_20230719-185419_format.csv`: Total number of inhabitants per Swedish county in 2018. Cleaned version of `BE0101N1_20230719-185419.csv`.

+ `active_population_2019_format.csv`: Number of Swedish inhabitants between 16-65 years old (active population) per county. Formatted from `data/interim/epi/demographic/age_structure_SWE_2019.csv`.

+ `recurrent_mobility_normtotal_SWE.csv`: contains the normalized recurrent mobility matrix between the Swedish counties for 2018. Recurrent mobility matrix `AM0207AC_20230705-141955_format.csv` normalised with the total population `BE0101N1_20230719-185419_format.csv`.

+ `recurrent_mobility_normactive_SWE.csv`: contains the normalized recurrent mobility matrix between the Swedish counties for 2018. Recurrent mobility matrix `AM0207AC_20230705-141955_format.csv` normalised with the total active population `active_population_2019_format.csv`.

### eco

#### policies

`policies_SWE.csv`: Contains the closure of schools for upper secundary and higher education during the 2020 COVID-19 lockdown, approximated as schools remaining 80% open. An overview of policies can be found in: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7537539/ 

`policies_BE.csv`: Contains the many Belgian economic policies during the 2020 COVID-19 pandemic. Retrieved from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8280466/ and https://www.sciencedirect.com/science/article/pii/S1755436521000542#appSB 

`policies_BE_scenarios.csv`: Four scenarios for economic shutdown: L1: Complete economic lockdown. L2: Closure of all B2C activities (a: schools closed, b: schools open). L3: Closure of sectors I, R, S94 only (a: schools closed, b: schools open), L4: No economic lockdown.

#### calibration_data

+ `SWE.csv`: Contains the nation-level GDP and unemployed fraction of the active population for Sweden. National-level GDP obtained from `am0401_2023m06_di_01_en_aku.xls`, unemployment data obtained from `am0401_2023m06_di_01_en_aku.xls`. 

+ `BE.csv`: Contains the nation-level GDP and unemployed fraction of the active population for Belgium. Copied from our COVID-19 project. Synthetic GDP is obtained from the statistics website of the Belgian National Bank (be.stat) while unemployment is proxied by the number of temporarily unemployed workers(survey by the ERMG).

#### labor_market_composition

+ `sector_structure_by_residence_SWE_2019.csv`: Number (and fraction of the county's population) of gainfully employed by Swedish county of residence (RAMS) and industrial classification NACE Rev. 2. Data for 2018. Cleaned version of `AM0207I5_20230705-155636.csv`. I verified the relative fractions in every county sum to one.
sector_structure_by_work_SWE_2019.csv

+ `sector_structure_by_work_SWE_2019_original.csv`: Number (and fraction of the county's total number of workers) of gainfully employed by Swedish county of work (RAMS) and industrial classification NACE Rev. 2. Data for 2018. Cleaned version of `AM0207I6_20230706-115023.csv`. I verified the relative fractions in every county sum to one.

+ `sector_structure_by_work_SWE.csv`: Number (and fraction of the county's total number of workers) of gainfully employed by Swedish county of work (RAMS) and industrial classification NACE Rev. 2. Data for 2018. Aggregated sectors (f.e. 'S, T') expanded to match the NACE 21 classification ('S' and 'T').  Aggregated sectors are disentangled using the national number of employees at the NACE 64 level (found in IO tables). Converted from: `sector_structure_by_work_SWE_2019_original.csv` using `expand_sector_structure_SWE.ipynb`.

+ `sector_structure_by_work_BE.csv`: Column 'abs' is a cleaned version of the data in `sector_structure_by_work_BE_raw.csv`. Column 'rel' is obtained by dividing by the total number of employed persons between 15-64 year old in Belgium.

+ `active_population_BE.csv`: Employed fraction 15-64 year olds. Population between 15-64 year. Total population. Per Belgian province. Census of 2011. Cleaned version of `active_population_BE_raw.csv`.

#### national_accounts

##### BE

+ `IO_BE_NACE64.csv`: symmetric input-output table for Belgium, formatted to NACE 64 classification starting from `vR64_nl_20191213.xlsx`. Values from tab Tbl_8 "Symmetrische input-outputtabel (product x product)". Sector L68 is split in two in the raw IO table: '68_' (real estate minus rent) and '68a' (rent). In the formatted IO table, rent is ommitted for the following reason: Under a pandemic shock households will not stop paying rent, either because they retain their income, or because the govnerment furloughs them (in either case people didn't get thrown out of their houses during the COVID-19 crisis). Ommitting rent from L68 implies that L68 in our model represents the buying-a-house on-site consumption type of real estate activities.

`other_accounts_BE_NACE64.csv`: All other variables from the Belgian national accounts `vR64_nl_20191213.xlsx` needed to initialize the model. 

##### SWE

`IO_SWE_NACE64.csv`: symmetric input-output table for Sweden in 2019, formatted to NACE 64 classification starting from `input_output_produkt_2015-2020_korr.xlsx` Similarily to to the Belgian IO matrix, revenue from rent was removed from L68 (delete corresponding row and column from matrix).

`other_accounts_SWE.csv`: All other variables from the Swedish national accounts `input_output_produkt_2015-2020_korr.xlsx` (2019 SIOT) needed to initialize the model. The total number of employees is sadly not listed in `input_output_produkt_2015-2020_korr.xlsx` (2019) but it is in `nrio_siot_181108.xlsx` (2016). However, some sectors are aggregated in `nrio_siot_181108.xlsx` (f.i. C20-21 instead of C20 and C21). I have used the fraction of employees derived from the Belgian IO Table to disaggregate these sectors in Sweden.

#### pichler

+ `IHS_critical_NACE64.csv` contains the IHS Market Analysts data (`IHS_Markit_results_compact.csv`), reformatted from WIOD 55 to the NACE 64 classification. Rows $i$ represent to what extent sectors $j$ depend on the goods of $i$ to produce good $j$. There are three dependencies: None (0), Important (0.5) and Critical (1). Critical dependecy of L68 (real estate) on H53 (Postal services) was removed. Dependency of H49 (Land Transport) and H51 (Air Transport) on I55-56 (Accodomodation) was removed because the closure of I55-56 during the lockdown led to overly large declines in economic activity. Sector Rental and Leasing (N77) critically depends on C33 (Repair of Machinery) and G45 (Retail of vehicles) only. Travel agencies (N79) critically depend on Air, Water and Land transport (H49/H50/H51) and importantly depend on I55-56, N77/N78. All dependencies of Public Administration (O84) and Education (P85) were relaxed as the government sector didn't face any shocks during the COVID-19 pandemic (https://www.nbb.be/doc/ts/publications/economicreview/2021/ecorevii2021.pdf). An "important" dependency of C29 (Production and repair of motor vehicles) and G45 (Retail of motor vehicles) on I55-56 was relaxed because it lowered the employment and GDP on the 2020 COVID-19 lockdown plateau in Belgium by over 10% too much (!).

+ `desired_stock_NACE64.csv`: For every sector in the NACE 64 classification, the number of days production can continue when no inputs are delivered (= stock). Converted version of `table_ratio_inv_go.csv`.

+ `on_site_consumption_NACE64.csv`: For every sector in the NACE 64 classification, if the consumption happens on-site.

+ `other_demand_shock.csv`: Other demand shock used by Pichler et. al. 

#### misc

+ `conversion_matrix_NACE64_NACE21.csv`:  contains conversion matrices to aggregate data from the NACE 64 to NACE 21 classification.

+ `conversion_matrix_NACE21_NACE10.csv`:  contains conversion matrices to aggregate data from NACE 21 to NACE 10.

+ `sector_labels_names.xlsx`: contains the full name of all NACE 64 sectors.