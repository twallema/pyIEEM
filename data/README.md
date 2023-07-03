# Description of datasets

## Raw

### epi

#### contacts

##### helicon

+ `figure1.jpg`: The 14-day incidence of COVID-19 in selected sectors in periods 29 September–12 October and 6–19 October. Figure 1 in "Verbeeck J, Vandersmissen G, Peeters J, Klamer S, Hancart S, Lernout T, Dewatripont M, Godderis L, Molenberghs G. Confirmed COVID-19 Cases per Economic Activity during Autumn Wave in Belgium. Int J Environ Res Public Health. 2021 Nov 27;18(23):12489."

+ `ijerph-1447404-supplementary.pdf`: Contains the 14-Day incidence of COVID-19 in sectors with minimum 10,000 employees at NACE Level 1/2 prior to the 19th October non-pharmaceutical interventions (Table S1/S2). Can be used to estimate the relative risk of infection in every stector of the economy. Retrieved from the supplementary materials of "Verbeeck J, Vandersmissen G, Peeters J, Klamer S, Hancart S, Lernout T, Dewatripont M, Godderis L, Molenberghs G. Confirmed COVID-19 Cases per Economic Activity during Autumn Wave in Belgium. Int J Environ Res Public Health. 2021 Nov 27;18(23):12489."

+ `incidences.xlsx`: Contains the extracted 14-Day incidence of COVID-19 in sectors with minimum 10,000 employees at NACE Level 1/2 prior to the 19th October non-pharmaceutical interventions (Table S1/S2 + Figure 1). Tab `20200929_20201012` contains incidences collected between Sep. 29, 2020 and Oct. 12, 2020, tab `20201006_20201019` contains incidences collected between Oct. 6, 2020 and Oct. 19, 2020.

##### matrices

Underlying folder structure ommitted from this readme. Contains the interaction matrix (in the place and time suggested by the folder and spreadsheets names) based on the 2008 study by Joel Mossong for Belgium and Finland. The spreadsheet has several tabs to distinguish between the nature and duration of the contact. Data extracted using the social contact rates data tool made by Lander Willem, available at https://lwillem.shinyapps.io/socrates_rshiny/. During extraction of the data, weighing by age, weighing by week/weekend were used and reciprocity was assumed. Both physical and non-physical contacts were included.

### eco

+ `employment_NACE64_2019.csv`: Number of employees per economic activity of NACE64 in Belgium (2019). Retrieved from NBB.stat > Population and Labour Market > Employment > Employment: annual detailed data > Domestic concept: A64.

## Interim

### epi

#### contacts

##### helicon

`incidences_NACE64.xlsx`: Contains the extracted 14-Day incidence of COVID-19 in sectors with minimum 10,000 employees at NACE Level 1/2 prior to the 19th October non-pharmaceutical interventions (Table S1/S2 + Figure 1), formatted using the raw data in `~data/raw/epi/contacts/helicon/incidences.xlsx` to match the NACE 64 economic activities. Normalised with the incidence over all sectors. The following assumptions were made during the conversion: B05-09, equal to C; C10-12, equal to average of C1011, C1012, C1013; N80-82: equal to average N80 and N82; Q87-88: equal to average of Q and Q87; R90-92: equal to average of R and R90; T97-98: equal to S96. Tab `20200929_20201012` contains incidences collected between Sep. 29, 2020 and Oct. 12, 2020, tab `20201006_20201019` contains incidences collected between Oct. 6, 2020 and Oct. 19, 2020. Tab `comparison` normalises the relative incidences so that the employee-weighted average relative incidence sums to one.

### eco

