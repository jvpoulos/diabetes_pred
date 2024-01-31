--Modified from Q:\Projects\DiabetesPredictiveModeling\SQL_examples\DataProcessingExamples\SampleCodeFromDiabetesOutcomesProjectV05.sql
              --Q:\Projects\DiabetesPredictiveModeling\SQL_examples\DataProcessingExamples\SnippetsforInclusionCriteria.sql
              --Q:\Projects\DiabetesPredictiveModeling\SQL_examples\DataProcessingExamples\CreateDeclineAnalyticalTable_Combined_v2.sql

--creates analysis table: DiabetesOutcomes

--------------------------------------------------------------------------------------------------------
--Defined values
--------------------------------------------------------------------------------------------------------
DECLARE @minEntryDate DATE     -- earliest accepted EntryDate
DECLARE @maxEntryDate DATE     -- latest accepted EntryDate 
DECLARE @minDate DATE          -- earliest logical date cutoff for criterion tables
DECLARE @maxDate DATE

DECLARE @min_A1c FLOAT
DECLARE @max_A1c FLOAT

SET @minEntryDate = '2000-01-01'
SET @maxEntryDate = '2022-06-30' 
SET @minDate = '1990-01-01'
SET @maxDate = '2022-12-31' 

SET @min_A1c = 4.0
SET @max_A1c = 20.0
DECLARE @min_A1c_inclusion FLOAT = 7.0

--------------------------------------------------------------------------------------------------------
--Inclusion and exlusion criteria (start with n=91557)
--------------------------------------------------------------------------------------------------------

--Identify Patients with At Least Two Encounters
IF OBJECT_ID('tempdb..#tmp_PrimCarePatients') IS NOT NULL
    DROP TABLE #tmp_PrimCarePatients;

SELECT a.EMPI, 
       COUNT(DISTINCT a.serviceDate) AS NumberEncounters, 
       MIN(a.serviceDate) AS FirstEncounterDate,  -- Store the first encounter date
       MAX(a.serviceDate) AS LastEncounterDate,  -- Store the last encounter date
       DATEADD(YEAR, 18, d.Date_of_Birth) AS Date18, 
       d.Date_of_Birth, 
       d.Gender_Legal_Sex AS Gender
INTO #tmp_PrimCarePatients
FROM (
    SELECT DISTINCT EMPI, DateofServiceDTS AS serviceDate
    FROM noteHeadersEpic
    WHERE DepartmentSpecialtyCD IN ('182', '9', '82', '12', '17', '125', '193')
        AND DateOfServiceDTS BETWEEN @minDate AND @maxDate
    UNION
    SELECT DISTINCT EMPI, serviceDate
    FROM noteHeadersLMR n
    JOIN ClinicSpecialtiesLMR c ON n.clinid = c.clinID AND primaryCare = '1' AND n.draftStatus = 'F'
    WHERE n.serviceDate BETWEEN @minDate AND @maxDate
) AS a
JOIN dem_2_pcp_combined d ON d.EMPI = a.EMPI
JOIN EMPIs2PrimaryCareEnc02 pc ON pc.empi = a.empi
GROUP BY a.EMPI, d.Date_of_Birth, d.Gender_Legal_Sex
HAVING COUNT(DISTINCT a.serviceDate) > 1; --n=90259
--SELECT TOP 50 PERCENT * FROM #tmp_PrimCarePatients

--------------------------------------------------------------------------------------------------------
--A1c data
--------------------------------------------------------------------------------------------------------

-- Clean A1c Values for those in primary care patients
IF OBJECT_ID('tempdb..#CleanedA1cAverages') IS NOT NULL
    DROP TABLE #CleanedA1cAverages;

SELECT e.EMPI, e.A1cDate, AVG(e.A1c) AS A1c
INTO #CleanedA1cAverages
FROM (
    SELECT l.EMPI, CAST(l.LabDate AS DATE) AS A1cDate, l.nval AS 'A1c'
    FROM labsEPIC l
    INNER JOIN #tmp_PrimCarePatients pc ON l.EMPI = pc.EMPI
    WHERE l.StudyLabCode LIKE '%A1c%' 
          AND ISNUMERIC(l.nval) > 0 
          AND l.nval >= @min_A1c AND l.nval <= @max_A1c
    UNION ALL
    SELECT lm.EMPI, CAST(lm.LabDate AS DATE) AS A1cDate, lm.nval AS 'A1c'
    FROM labsLMR lm
    INNER JOIN #tmp_PrimCarePatients pc ON lm.EMPI = pc.EMPI
    WHERE lm.GroupCD LIKE '%A1c%'
          AND ISNUMERIC(lm.nval) > 0 
          AND lm.nval >= @min_A1c AND lm.nval <= @max_A1c
    UNION ALL
    SELECT la.EMPI, CAST(la.LabDate AS DATE) AS A1cDate, la.nval AS 'A1c'
    FROM labsLMRArchive la
    INNER JOIN #tmp_PrimCarePatients pc ON la.EMPI = pc.EMPI
    WHERE la.GroupCD LIKE '%A1c%' 
          AND ISNUMERIC(la.nval) > 0 
          AND la.nval >= @min_A1c AND la.nval <= @max_A1c
) AS e
GROUP BY e.EMPI, e.A1cDate; --n=1922522
--SELECT TOP 50 PERCENT * FROM #CleanedA1cAverages

--HbA1c ≥ 7.0% during the study period

-- Create a temporary table to store dates of elevated A1c along with additional columns
IF OBJECT_ID('tempdb..#tmp_A1cElevated') IS NOT NULL
    DROP TABLE #tmp_A1cElevated;

CREATE TABLE #tmp_A1cElevated (
    EMPI VARCHAR(50),
    A1cDate DATE,
    Age INT,
    Gender VARCHAR(10),
    nval FLOAT
);

-- Insert into #tmp_A1cElevated from #CleanedA1cAverages
INSERT INTO #tmp_A1cElevated (EMPI, A1cDate, Age, Gender, nval)
SELECT DISTINCT 
    ca.EMPI, 
    ca.A1cDate, 
    DATEDIFF(year, p.Date_of_Birth, ca.A1cDate) AS Age, 
    p.Gender_Legal_Sex AS Gender,
    ca.A1c
FROM #CleanedA1cAverages ca
JOIN dem_2_pcp_combined p ON p.EMPI = ca.EMPI
WHERE ca.A1c >= @min_A1c_inclusion
  AND ca.A1cDate > @minEntryDate; --n=977412
--SELECT TOP 50 PERCENT * FROM #tmp_A1cElevated;

-- Determine Eligible Periods and Index Date
--1. Identify the latest date among the patient turning 18, the study start date, and the first PCP note for each patient.
--2. Find the most recent A1c measurement greater than or equal to 7 before LatestDate and after @minDate.
--3. If no such A1c measurement is found before the LatestDate, find the next A1c measurement greater than or equal to 7.
--4. The study entry date (IndexDate) is the latest date among the LatestDate and A1cDate.
--5. Ensure that the search for an IndexDate does not go beyond @maxEntryDate or LastEncounterDate.

IF OBJECT_ID('tempdb..#PatientConditions') IS NOT NULL
    DROP TABLE #PatientConditions;

IF OBJECT_ID('tempdb..#A1cMeasurements') IS NOT NULL
    DROP TABLE #A1cMeasurements;

IF OBJECT_ID('tempdb..#tmp_indexDate') IS NOT NULL
    DROP TABLE #tmp_indexDate;

-- Create temporary tables
SELECT 
    p.EMPI, 
    DATEADD(YEAR, 18, p.Date_of_Birth) AS PatientTurns18,
    p.FirstEncounterDate,
    p.LastEncounterDate,
    CASE 
        WHEN DATEADD(YEAR, 18, p.Date_of_Birth) > p.FirstEncounterDate AND DATEADD(YEAR, 18, p.Date_of_Birth) > @minEntryDate THEN DATEADD(YEAR, 18, p.Date_of_Birth)
        WHEN p.FirstEncounterDate > @minEntryDate THEN p.FirstEncounterDate
        ELSE @minEntryDate
    END AS LatestDate
INTO #PatientConditions
FROM #tmp_PrimCarePatients p;
CREATE INDEX idx_patient ON #PatientConditions(EMPI); --n=90259
--SELECT * FROM #PatientConditions; 

SELECT
    a.EMPI,
    a.A1cDate,
    a.nval
INTO #A1cMeasurements
FROM #tmp_A1cElevated a
WHERE a.nval >= @min_A1c_inclusion AND a.A1cDate BETWEEN @minDate AND @maxEntryDate;
CREATE INDEX idx_a1c ON #A1cMeasurements(EMPI, A1cDate); --n=888967
--SELECT * FROM #A1cMeasurements; 

-- Determine IndexDate, A1cValue, and A1cDate
WITH A1cData AS (
    SELECT 
        am.EMPI,
        am.A1cDate,
        am.nval AS A1cValue,
        ROW_NUMBER() OVER (
            PARTITION BY am.EMPI 
            ORDER BY 
                CASE 
                    WHEN am.A1cDate <= pc.LatestDate AND am.nval >= 7 THEN 0 
                    ELSE 1 
                END,
                ABS(DATEDIFF(DAY, am.A1cDate, pc.LatestDate))
        ) AS rn
    FROM #A1cMeasurements am
    INNER JOIN #PatientConditions pc ON am.EMPI = pc.EMPI
    WHERE (am.A1cDate <= pc.LatestDate AND am.nval >= 7) OR 
          (am.A1cDate > pc.LatestDate AND am.nval >= 7 AND am.A1cDate <= pc.LastEncounterDate)
)
SELECT 
    pc.EMPI,
    pc.PatientTurns18,
    pc.FirstEncounterDate,
    pc.LastEncounterDate,
    pc.LatestDate,
    CASE 
        WHEN ad.A1cDate IS NOT NULL THEN 
            CASE 
                WHEN ad.A1cDate > pc.LatestDate THEN 
                    CASE 
                        WHEN ad.A1cDate <= @maxEntryDate AND ad.A1cDate <= pc.LastEncounterDate THEN ad.A1cDate 
                        ELSE pc.LatestDate 
                    END
                ELSE pc.LatestDate 
            END
        ELSE pc.LatestDate
    END AS IndexDate,
    ad.A1cValue,
    ad.A1cDate
INTO #tmp_indexDate
FROM #PatientConditions pc
LEFT JOIN A1cData ad ON pc.EMPI = ad.EMPI AND ad.rn = 1
WHERE ad.A1cValue IS NOT NULL --n=76460

-- Delete 431 rows where IndexDate exceeds @maxEntryDate or LastEncounterDate
DELETE FROM #tmp_indexDate
WHERE 
    (IndexDate > @maxEntryDate) OR 
    (IndexDate > LastEncounterDate)

SELECT * FROM #tmp_indexDate --n =76029
--WHERE A1cDate = LatestDate --n = 3925
--WHERE A1cDate > LatestDate --n = 44223
--WHERE A1cDate < LatestDate --n = 27881
--WHERE A1cDate < IndexDate --n = 27881
--WHERE A1cDate = IndexDate --n = 48148
--------------------------------------------------------------------------------------------------------
--Additional exclusion criteria
--------------------------------------------------------------------------------------------------------

--Exclude those whose age is < 18 years or > 75 years
IF OBJECT_ID('tempdb..#tmp_Under75') IS NOT NULL
    DROP TABLE #tmp_Under75;

SELECT 
    id.EMPI,
    id.IndexDate,
    DATEDIFF(YEAR, d.Date_of_Birth, id.IndexDate) AS AgeYears,
    d.Gender_Legal_Sex
INTO #tmp_Under75
FROM #tmp_indexDate id
INNER JOIN dem_2_pcp_combined d ON id.EMPI = d.EMPI
WHERE DATEDIFF(YEAR, d.Date_of_Birth, id.IndexDate) BETWEEN 18 AND 75; --n=64614

--Exclude those with unknown gender
IF OBJECT_ID('tempdb..#tmp_studyPop') IS NOT NULL
    DROP TABLE #tmp_studyPop;

SELECT *
INTO #tmp_studyPop
FROM #tmp_Under75
WHERE Gender_Legal_Sex IN ('Female', 'Male'); --n=64611
--SELECT * FROM #tmp_studyPop

--------------------------------------------------------------------------------------------------------
--Identify hyperglycemic periods
--------------------------------------------------------------------------------------------------------

IF OBJECT_ID('tempdb..#A1c12MonthsLaterTable') IS NOT NULL
    DROP TABLE #A1c12MonthsLaterTable;

-- Use A1cValueAtIndexDate from #tmp_indexDate
WITH InitialA1c AS (
    SELECT 
        idx.EMPI,
        idx.IndexDate,
        idx.A1cDate AS InitialA1cDate,
        idx.A1cValue AS InitialA1c
    FROM #tmp_indexDate idx
),

-- Find the A1c measurement approximately 12 months later
A1c12MonthsLater AS (
    SELECT 
        i.EMPI,
        i.IndexDate,
        i.InitialA1cDate,
        i.InitialA1c,
        ca.A1cDate AS A1cDateAfter12Months,
        ca.A1c AS A1cAfter12Months,
        ROW_NUMBER() OVER (PARTITION BY i.EMPI ORDER BY ABS(DATEDIFF(MONTH, i.InitialA1cDate, ca.A1cDate) - 12)) AS rn
    FROM InitialA1c i
    INNER JOIN #CleanedA1cAverages ca ON i.EMPI = ca.EMPI
    WHERE ca.A1cDate > i.InitialA1cDate
    AND ca.A1cDate <= DATEADD(MONTH, 15, i.InitialA1cDate)
) --n=68319

-- Final selection into a temporary table
SELECT
    a.EMPI,
    a.IndexDate,
    a.InitialA1cDate,
    a.InitialA1c,
    a.A1cDateAfter12Months,
    a.A1cAfter12Months,
    CASE WHEN a.A1cAfter12Months >= 7 THEN 1 ELSE 0 END AS A1cGreaterThan7
INTO #A1c12MonthsLaterTable
FROM A1c12MonthsLater a
WHERE a.rn = 1;

SELECT * FROM #A1c12MonthsLaterTable; --n=68319

--------------------------------------------------------------------------------------------------------
--Patient characteristics
--------------------------------------------------------------------------------------------------------

-- Add columns to #A1c12MonthsLaterTable if they don't exist
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoGender') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoGender varchar(50);
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoFemale') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoFemale int;
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoMarital') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoMarital varchar(50);
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoMarried') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoMarried int;
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoGovIns') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoGovIns int;
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoEnglish') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoEnglish int;

-- Update demographic data
UPDATE a
SET a.demoGender = d.Gender_Legal_Sex,
    a.demoFemale = CASE WHEN d.Gender_Legal_Sex = 'Female' THEN 1 ELSE 0 END,
    a.demoEnglish = CASE WHEN d.Language LIKE 'ENG%' THEN 1 ELSE 0 END,
    a.demoMarital = CASE 
                        WHEN d.Marital_status IN ('Widow', 'Law', 'Partner', 'Married') THEN 'Married/Partnered'
                        WHEN d.Marital_status IN ('Separated', 'Divorce') THEN 'Separated'
                        WHEN d.Marital_status = 'Single' THEN 'Single'
                        ELSE 'Unknown'
                    END,
    a.demoMarried = CASE WHEN d.Marital_status IN ('Married', 'Law', 'Partner') THEN 1 ELSE 0 END
FROM #A1c12MonthsLaterTable a
JOIN dem_2_pcp_combined d ON a.EMPI = d.EMPI; --n=68319

-- Update government insurance data
UPDATE a
SET a.demoGovIns = CASE 
                    WHEN c.insurance_1 LIKE '%MEDICARE%' OR c.insurance_2 LIKE '%MEDICARE%' OR c.insurance_3 LIKE '%MEDICARE%' THEN 1
                    WHEN c.insurance_1 LIKE '%MEDICAID%' OR c.insurance_2 LIKE '%MEDICAID%' OR c.insurance_3 LIKE '%MEDICAID%' THEN 1
                    WHEN c.insurance_1 LIKE '%MASSHEALTH%' OR c.insurance_2 LIKE '%MASSHEALTH%' OR c.insurance_3 LIKE '%MASSHEALTH%' THEN 1
                    ELSE 0
                END
FROM #A1c12MonthsLaterTable a
JOIN con_2_pcp_combined c ON a.EMPI = c.EMPI;

SELECT * FROM #A1c12MonthsLaterTable; --n=68319

--------------------------------------------------------------------------------------------------------
--Create dataset
--------------------------------------------------------------------------------------------------------
--initialize the DiabetesOutcomes table directly from #A1c12MonthsLaterTable
--include relevant columns from #tmp_studyPop and #tmp_PrimCarePatients

IF OBJECT_ID('dbo.DiabetesOutcomes', 'U') IS NOT NULL
    DROP TABLE dbo.DiabetesOutcomes;

-- Create the DiabetesOutcomes table with a new study-specific ID
-- Keep EMPI for now
SELECT 
    ROW_NUMBER() OVER (ORDER BY a12.EMPI) AS newID, -- Generate a unique identifier
    a12.EMPI,
    a12.InitialA1c,
    a12.A1cAfter12Months,
    a12.A1cGreaterThan7,
    a12.demoFemale,
    a12.demoMarried,
    a12.demoGovIns,
    a12.demoEnglish,
    DATEDIFF(DAY, id.IndexDate, a12.InitialA1cDate) AS DaysFromIndexToInitialA1cDate,
    DATEDIFF(DAY, id.IndexDate, a12.A1cDateAfter12Months) AS DaysFromIndexToA1cDateAfter12Months,
    DATEDIFF(DAY, id.IndexDate, id.FirstEncounterDate) AS DaysFromIndexToFirstEncounterDate,
    DATEDIFF(DAY, id.IndexDate, id.LastEncounterDate) AS DaysFromIndexToLastEncounterDate,
    DATEDIFF(DAY, id.IndexDate, id.LatestDate) AS DaysFromIndexToLatestDate,
    DATEDIFF(DAY, id.IndexDate, id.PatientTurns18) AS DaysFromIndexToPatientTurns18,
   -- id.IndexDate, -- Keeping the original IndexDate for reference
    sp.AgeYears, -- AgeYears from #tmp_studyPop
    ppc.NumberEncounters -- NumberEncounters from #tmp_PrimCarePatients
INTO dbo.DiabetesOutcomes
FROM #A1c12MonthsLaterTable a12
INNER JOIN #tmp_indexDate id ON a12.EMPI = id.EMPI
INNER JOIN #tmp_studyPop sp ON a12.EMPI = sp.EMPI
INNER JOIN #tmp_PrimCarePatients ppc ON a12.EMPI = ppc.EMPI;--n=55667
SELECT * FROM dbo.DiabetesOutcomes;


-- Merge columns from selected RDPR tables into DiabetesOutcomes
 /*
IF OBJECT_ID('dbo.DiabetesOutcomes', 'U') IS NOT NULL
    DROP TABLE dbo.DiabetesOutcomes;

-- Create the DiabetesOutcomes table
SELECT 
    a12.EMPI AS Dataset_EMPI,
    a12.InitialA1cDate AS ElevatedA1cDate,
    id.LatestIndexDate AS IndexDate,
    a12.A1cDateAfter12Months,
    a12.InitialA1c,
    a12.A1cAfter12Months, --n=46157

    /*
    -- Columns from all_2_pcp_combined with Allp_ prefix
    allp.EPIC_PMRN AS Allp_EPIC_PMRN,
    allp.MRN_Type AS Allp_MRN_Type,
    allp.MRN AS Allp_MRN,
    allp.System AS Allp_System,
    allp.Noted_Date AS Allp_Noted_Date,
    allp.Allergen AS Allp_Allergen,
    allp.Allergen_Type AS Allp_Allergen_Type,
    allp.Allergen_Code AS Allp_Allergen_Code,
    allp.Reactions AS Allp_Reactions,
    allp.Severity AS Allp_Severity,
    allp.Reaction_Type AS Allp_Reaction_Type,
    allp.Comments AS Allp_Comments,
    allp.Status AS Allp_Status,
    allp.Deleted_Reason_Comments AS Allp_Deleted_Reason_Comments,
    */

    -- Columns from con_2_pcp_combined with Conp_ prefix
    conp.EPIC_PMRN AS Conp_EPIC_PMRN,
    conp.MRN_Type AS Conp_MRN_Type,
    conp.MRN AS Conp_MRN,
    conp.Last_Name AS Conp_Last_Name,
    conp.First_Name AS Conp_First_Name,
    conp.Middle_Name AS Conp_Middle_Name,
    conp.Research_Invitations AS Conp_Research_Invitations,
    conp.Address1 AS Conp_Address1,
    conp.Address2 AS Conp_Address2,
    conp.City AS Conp_City,
    conp.State AS Conp_State,
    conp.Zip AS Conp_Zip,
    conp.Country AS Conp_Country,
    conp.Home_Phone AS Conp_Home_Phone,
    conp.Day_Phone AS Conp_Day_Phone,
    conp.SSN AS Conp_SSN,
    conp.VIP AS Conp_VIP,
    conp.Previous_Name AS Conp_Previous_Name,
    conp.Patient_ID_List AS Conp_Patient_ID_List,
    conp.Insurance_1 AS Conp_Insurance_1,
    conp.Insurance_2 AS Conp_Insurance_2,
    conp.Insurance_3 AS Conp_Insurance_3,
    conp.Primary_Care_Physician AS Conp_Primary_Care_Physician,
    conp.Resident_Primary_Care_Physician AS Conp_Resident_Primary_Care_Physician,
    
    -- Columns from dem_2_pcp_combined with Demp_ prefix
    demp.EPIC_PMRN AS Demp_EPIC_PMRN,
    demp.MRN_Type AS Demp_MRN_Type,
    demp.MRN AS Demp_MRN,
    demp.Gender_Legal_Sex AS Demp_Gender_Legal_Sex,
    demp.Date_of_Birth AS Demp_Date_of_Birth,
    demp.Age AS Demp_Age,
    demp.Sex_At_Birth AS Demp_Sex_At_Birth,
    demp.Gender_Identity AS Demp_Gender_Identity,
    demp.Language AS Demp_Language,
    demp.Language_group AS Demp_Language_group,
    demp.Race1 AS Demp_Race1,
    demp.Race2 AS Demp_Race2,
    demp.Race_Group AS Demp_Race_Group,
    demp.Ethnic_Group AS Demp_Ethnic_Group,
    demp.Marital_status AS Demp_Marital_status,
    demp.Religion AS Demp_Religion,
    demp.Is_a_veteran AS Demp_Is_a_veteran,
    demp.Zip_code AS Demp_Zip_code,
    demp.Country AS Demp_Country,
    demp.Vital_status AS Demp_Vital_status,
    demp.Date_Of_Death AS Demp_Date_Of_Death

    /*
    -- Columns from dia_2_pcp_combined with Diap_ prefix
    diap.EPIC_PMRN AS Diap_EPIC_PMRN,
    diap.MRN_Type AS Diap_MRN_Type,
    diap.MRN AS Diap_MRN,
    diap.Date AS Diap_Date,
    diap.Diagnosis_Name AS Diap_Diagnosis_Name,
    diap.Code_Type AS Diap_Code_Type,
    diap.Code AS Diap_Code,
    diap.Diagnosis_Flag AS Diap_Diagnosis_Flag,
    diap.Provider AS Diap_Provider,
    diap.Clinic AS Diap_Clinic,
    diap.Hospital AS Diap_Hospital,
    diap.Inpatient_Outpatient AS Diap_Inpatient_Outpatient,
    diap.Encounter_number AS Diap_Encounter_number

    -- Columns from enc_2_pcp_combined with Encp_ prefix
    encp.EPIC_PMRN AS Encp_EPIC_PMRN,
    encp.MRN_Type AS Encp_MRN_Type,
    encp.MRN AS Encp_MRN,
    encp.Encounter_number AS Encp_Encounter_number,
    encp.Encounter_Status AS Encp_Encounter_Status,
    encp.Hospital AS Encp_Hospital,
    encp.Inpatient_Outpatient AS Encp_Inpatient_Outpatient,
    encp.Service_Line AS Encp_Service_Line,
    encp.Attending_MD AS Encp_Attending_MD,
    encp.Admit_Date AS Encp_Admit_Date,
    encp.Discharge_Date AS Encp_Discharge_Date,
    encp.LOS_Days AS Encp_LOS_Days,
    encp.Clinic_Name AS Encp_Clinic_Name,
    encp.Admit_Source AS Encp_Admit_Source,
    encp.Discharge_Disposition AS Encp_Discharge_Disposition,
    encp.Payor AS Encp_Payor,
    encp.Admitting_Diagnosis AS Encp_Admitting_Diagnosis,
    encp.Principal_Diagnosis AS Encp_Principal_Diagnosis,
    encp.Diagnosis_1 AS Encp_Diagnosis_1,
    encp.Diagnosis_2 AS Encp_Diagnosis_2,
    encp.Diagnosis_3 AS Encp_Diagnosis_3,
    encp.Diagnosis_4 AS Encp_Diagnosis_4,
    encp.Diagnosis_5 AS Encp_Diagnosis_5,
    encp.Diagnosis_6 AS Encp_Diagnosis_6,
    encp.Diagnosis_7 AS Encp_Diagnosis_7,
    encp.Diagnosis_8 AS Encp_Diagnosis_8,
    encp.Diagnosis_9 AS Encp_Diagnosis_9,
    encp.Diagnosis_10 AS Encp_Diagnosis_10,
    encp.DRG AS Encp_DRG,
    encp.Patient_Type AS Encp_Patient_Type,
    encp.Referrer_Discipline AS Encp_Referrer_Discipline,

    -- Columns from phy_2_pcp_combined with Phyp_ prefix
    phyp.EPIC_PMRN AS Phyp_EPIC_PMRN,
    phyp.MRN_Type AS Phyp_MRN_Type,
    phyp.MRN AS Phyp_MRN,
    phyp.Date AS Phyp_Date,
    phyp.Concept_Name AS Phyp_Concept_Name,
    phyp.Code_Type AS Phyp_Code_Type,
    phyp.Code AS Phyp_Code,
    phyp.Result AS Phyp_Result,
    phyp.Units AS Phyp_Units,
    phyp.Provider AS Phyp_Provider,
    phyp.Clinic AS Phyp_Clinic,
    phyp.Hospital AS Phyp_Hospital,
    phyp.Inpatient_Outpatient AS Phyp_Inpatient_Outpatient,
    phyp.Encounter_number AS Phyp_Encounter_number,

    -- Columns from prc_2_pcp_combined with Prcp_ prefix
    prcp.EPIC_PMRN AS Prcp_EPIC_PMRN,
    prcp.MRN_Type AS Prcp_MRN_Type,
    prcp.MRN AS Prcp_MRN,
    prcp.Date AS Prcp_Date,
    prcp.Procedure_Name AS Prcp_Procedure_Name,
    prcp.Code_Type AS Prcp_Code_Type,
    prcp.Code AS Prcp_Code,
    prcp.Procedure_Flag AS Prcp_Procedure_Flag,
    prcp.Quantity AS Prcp_Quantity,
    prcp.Provider AS Prcp_Provider,
    prcp.Clinic AS Prcp_Clinic,
    prcp.Hospital AS Prcp_Hospital,
    prcp.Inpatient_Outpatient AS Prcp_Inpatient_Outpatient,
    prcp.Encounter_number AS Prcp_Encounter_number
    */
   -- mrnp.*, 
 --   labse.*, 
  --  labsl.*, 
  --  labsa.*, 
 --   medse.*, 
 --   medsl.*, 
 --   cs.*, 
 --   nhe.*, 
--    nhl.*
INTO dbo.DiabetesOutcomes
FROM #A1c12MonthsLaterTable a12
LEFT JOIN con_2_pcp_combined conp ON a12.EMPI = conp.EMPI
LEFT JOIN dem_2_pcp_combined demp ON a12.EMPI = demp.EMPI
LEFT JOIN (
    SELECT 
        EMPI,
        MAX(IndexDate) AS LatestIndexDate
    FROM #tmp_indexDate
    GROUP BY EMPI
) id ON a12.EMPI = id.EMPI; --n=46157
--LEFT JOIN dia_2_pcp_combined diap ON a12.EMPI = diap.EMPI
--LEFT JOIN enc_2_pcp_combined encp ON a12.EMPI = encp.EMPI
--LEFT JOIN phy_2_pcp_combined phyp ON a12.EMPI = phyp.EMPI
--LEFT JOIN prc_2_pcp_combined prcp ON a12.EMPI = prcp.EMPI
--LEFT JOIN mrn_2_pcp_combined mrnp ON a12.EMPI = mrnp.EMPI
--LEFT JOIN labsEPIC labse ON a12.EMPI = labse.EMPI
--LEFT JOIN labsLMR labsl ON a12.EMPI = labsl.EMPI
--LEFT JOIN labsLMRArchive labsa ON a12.EMPI = labsa.EMPI
--LEFT JOIN medsEpic medse ON a12.EMPI = medse.EMPI
--LEFT JOIN medsLMR medsl ON a12.EMPI = medsl.EMPI
--LEFT JOIN ClinicSpecialtiesLMR cs ON a12.EMPI = cs.EMPI
--LEFT JOIN noteHeadersEpic nhe ON a12.EMPI = nhe.EMPI
--LEFT JOIN noteHeadersLMR nhl ON a12.EMPI = nhl.EMPI;

 */