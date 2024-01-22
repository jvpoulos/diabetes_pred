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
--SELECT * FROM #tmp_PrimCarePatients

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
--1. Calculate the latest of the three dates (patient turns 18, study start date, first PCP note) for each patient.
--2. Identify the most recent A1c measurement prior to this calculated date.
--3. If this A1c measurement is less than 7, find the next date where it is greater than or equal to 7.
--4. Ensure that the search for an IndexDate does not go beyond @maxEntryDate and the last PCP encounter date.
--5. Guarantee that each patient has an IndexDate and avoid NULL values.

IF OBJECT_ID('tempdb..#tmp_indexDate') IS NOT NULL
    DROP TABLE #tmp_indexDate;

-- Calculate the latest of three conditions for each patient
WITH PatientConditions AS (
    SELECT 
        p.EMPI, 
        p.Date_of_Birth, 
        p.FirstEncounterDate,
        -- Calculate the latest of the three dates
        CASE 
            WHEN DATEADD(YEAR, 18, p.Date_of_Birth) > p.FirstEncounterDate AND DATEADD(YEAR, 18, p.Date_of_Birth) > @minEntryDate THEN DATEADD(YEAR, 18, p.Date_of_Birth)
            WHEN p.FirstEncounterDate > @minEntryDate THEN p.FirstEncounterDate
            ELSE @minEntryDate
        END AS LatestDate
    FROM #tmp_PrimCarePatients p
),
IndexDates AS (
    SELECT 
        pc.EMPI,
        COALESCE(
            -- Try to find the most recent A1c measurement before the LatestDate
            (
                SELECT TOP 1 a.A1cDate
                FROM #tmp_A1cElevated a
                WHERE a.EMPI = pc.EMPI
                  AND a.A1cDate < pc.LatestDate
                  AND a.nval >= @min_A1c_inclusion
                ORDER BY a.A1cDate DESC
            ),
            -- If not found, use the LatestDate as a fallback IndexDate
            pc.LatestDate
        ) AS IndexDate
    FROM PatientConditions pc
)

-- Store the result in #tmp_indexDate
SELECT * 
INTO #tmp_indexDate
FROM IndexDates; --n=90259
--SELECT * FROM #tmp_indexDate

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
WHERE DATEDIFF(YEAR, d.Date_of_Birth, id.IndexDate) BETWEEN 18 AND 75; --n=79837

--Exclude those with unknown gender
IF OBJECT_ID('tempdb..#tmp_studyPop') IS NOT NULL
    DROP TABLE #tmp_studyPop;

SELECT *
INTO #tmp_studyPop
FROM #tmp_Under75
WHERE Gender_Legal_Sex IN ('Female', 'Male'); --n=79833
--SELECT * FROM #tmp_studyPop

--------------------------------------------------------------------------------------------------------
--Identify hyperglycemic periods
--------------------------------------------------------------------------------------------------------

IF OBJECT_ID('tempdb..#A1c12MonthsLaterTable') IS NOT NULL
    DROP TABLE #A1c12MonthsLaterTable;

-- Step 1: Identify the first elevated A1c measurement for each patient in #tmp_studyPop using #tmp_A1cElevated
WITH FirstElevatedA1c AS (
    SELECT 
        ae.EMPI, 
        ae.A1cDate AS InitialA1cDate,
        ae.nval AS InitialA1c,
        ROW_NUMBER() OVER (PARTITION BY ae.EMPI ORDER BY ae.A1cDate) AS RowNum
    FROM #tmp_A1cElevated ae
    INNER JOIN #tmp_studyPop s ON ae.EMPI = s.EMPI
    WHERE ae.nval >= 7
    AND ae.A1cDate <= @maxEntryDate -- Ensuring the date is within the study period
),

-- Step 2: Find the A1c measurement approximately 12 months later
A1c12MonthsLater AS (
    SELECT 
        f.EMPI,
        f.InitialA1cDate,
        f.InitialA1c,
        MIN(ca.A1cDate) AS A1cDateAfter12Months,
        MIN(ca.A1c) AS A1cAfter12Months
    FROM FirstElevatedA1c f
    INNER JOIN #CleanedA1cAverages ca ON f.EMPI = ca.EMPI
        AND ca.A1cDate >= DATEADD(month, 9, f.InitialA1cDate) -- At least 9 months after
        AND ca.A1cDate <= DATEADD(month, 15, f.InitialA1cDate) -- No more than 15 months after
    WHERE f.RowNum = 1
    GROUP BY f.EMPI, f.InitialA1cDate, f.InitialA1c
)

-- Final selection into a temporary table
SELECT
    a.EMPI,
    a.InitialA1cDate,
    a.InitialA1c,
    a.A1cDateAfter12Months,
    a.A1cAfter12Months,
    CASE WHEN a.A1cAfter12Months >= 7 THEN 1 ELSE 0 END AS A1cGreaterThan7
INTO #A1c12MonthsLaterTable
FROM A1c12MonthsLater a; --n=45711
--SELECT TOP 50 PERCENT * FROM #A1c12MonthsLaterTable


--------------------------------------------------------------------------------------------------------
--Patient characteristics
--------------------------------------------------------------------------------------------------------

-- Add necessary columns to #A1c12MonthsLaterTable if they don't exist
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoMarital') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoMarital varchar(50);
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoMarried') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoMarried int;
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoGovIns') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoGovIns int;
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoGender') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoGender varchar(50);
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoFemale') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoFemale int;
IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoEnglish') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD demoEnglish int;

-- Create a temporary table for gender updates
IF OBJECT_ID('tempdb..#tempGenderUpdates') IS NOT NULL
    DROP TABLE #tempGenderUpdates;

SELECT 
    a.EMPI, 
    d.Gender_Legal_Sex AS demoGender,
    CASE WHEN d.Gender_Legal_Sex = 'Female' THEN 1 ELSE 0 END AS demoFemale
INTO #tempGenderUpdates
FROM #A1c12MonthsLaterTable a
JOIN dem_2_pcp_combined d ON a.EMPI = d.EMPI; --n=45711

-- Merge the gender data back into #A1c12MonthsLaterTable
MERGE INTO #A1c12MonthsLaterTable AS Target
USING #tempGenderUpdates AS Source
ON Target.EMPI = Source.EMPI
WHEN MATCHED THEN 
    UPDATE SET 
        Target.demoGender = Source.demoGender,
        Target.demoFemale = Source.demoFemale;

-- Categorize patient primary language
IF OBJECT_ID('tempdb..#tmp_demoLanguage') IS NOT NULL
    DROP TABLE #tmp_demoLanguage;

SELECT
    #A1c12MonthsLaterTable.EMPI,
    dem_2_pcp_combined.Language
INTO #tmp_demoLanguage
FROM #A1c12MonthsLaterTable
JOIN dem_2_pcp_combined ON #A1c12MonthsLaterTable.EMPI = dem_2_pcp_combined.EMPI; --n=45711

ALTER TABLE #tmp_demoLanguage
ADD demoEnglish int;

UPDATE #tmp_demoLanguage
SET demoEnglish = CASE
    WHEN Language LIKE 'ENG%' THEN 1
    ELSE 0
END;

MERGE INTO #A1c12MonthsLaterTable
USING #tmp_demoLanguage
ON #A1c12MonthsLaterTable.EMPI = #tmp_demoLanguage.EMPI
WHEN MATCHED THEN
    UPDATE SET #A1c12MonthsLaterTable.demoEnglish = #tmp_demoLanguage.demoEnglish;

-- Categorize patient marital status
-- Ensure that #tmp_demoMarital is dropped if it exists
IF OBJECT_ID('tempdb..#tmp_demoMarital') IS NOT NULL
    DROP TABLE #tmp_demoMarital;

-- Create the #tmp_demoMarital table
SELECT
    #A1c12MonthsLaterTable.EMPI,
    dem_2_pcp_combined.Marital_status
INTO #tmp_demoMarital
FROM #A1c12MonthsLaterTable
JOIN dem_2_pcp_combined ON #A1c12MonthsLaterTable.EMPI = dem_2_pcp_combined.EMPI; --n=45711

ALTER TABLE #tmp_demoMarital
ADD maritalStatusGroup varchar(50);

UPDATE #tmp_demoMarital
SET maritalStatusGroup = CASE
    WHEN Marital_status LIKE '%Widow%' THEN 'Widowed'
    WHEN Marital_status LIKE '%Law%' THEN 'Married/Partnered'
    WHEN Marital_status LIKE '%Partner%' THEN 'Married/Partnered'
    WHEN Marital_status LIKE '%Married%' THEN 'Married/Partnered'
    WHEN Marital_status LIKE '%Separated%' THEN 'Separated'
    WHEN Marital_status LIKE '%Divorce%' THEN 'Separated'
    WHEN Marital_status LIKE '%Single%' THEN 'Single'
    ELSE 'Unknown'
END;

MERGE INTO #A1c12MonthsLaterTable
USING #tmp_demoMarital
ON #A1c12MonthsLaterTable.EMPI = #tmp_demoMarital.EMPI
WHEN MATCHED THEN
    UPDATE SET #A1c12MonthsLaterTable.demoMarital = #tmp_demoMarital.maritalStatusGroup;

-- Ensure the #A1c12MonthsLaterTable exists and add demoMarried column if it doesn't exist
IF OBJECT_ID('#A1c12MonthsLaterTable', 'U') IS NOT NULL
BEGIN
    IF COL_LENGTH('#A1c12MonthsLaterTable', 'demoMarried') IS NULL
    BEGIN
        ALTER TABLE #A1c12MonthsLaterTable ADD demoMarried int;
    END;
END;

-- Now update the demoMarried column
UPDATE #A1c12MonthsLaterTable
SET demoMarried = CASE
    WHEN demoMarital = 'Married/Partnered' THEN 1
    ELSE 0
END;

UPDATE #A1c12MonthsLaterTable
SET demoFemale = 1 WHERE demoGender = 'Female'; --n=45711

-- Categorize patient insurance information

-- Drop the #tmp_demoInsurance table if it exists
IF OBJECT_ID('tempdb..#tmp_demoInsurance') IS NOT NULL
    DROP TABLE #tmp_demoInsurance;

-- Create the #tmp_demoInsurance table
SELECT 
    a.EMPI, 
    c.insurance_1, 
    c.insurance_2, 
    c.insurance_3
INTO #tmp_demoInsurance
FROM #A1c12MonthsLaterTable a
JOIN con_2_pcp_combined c ON a.EMPI = c.EMPI; --n=45711

-- Add governmentInsurance column to #tmp_demoInsurance
ALTER TABLE #tmp_demoInsurance
ADD governmentInsurance int;

-- Update #tmp_demoInsurance with governmentInsurance values
UPDATE #tmp_demoInsurance
SET governmentInsurance = 
    CASE
        WHEN insurance_1 LIKE '%MEDICARE%' OR
             insurance_2 LIKE '%MEDICARE%' OR
             insurance_3 LIKE '%MEDICARE%' THEN 1
        WHEN insurance_1 LIKE '%MEDICAID%' OR
             insurance_2 LIKE '%MEDICAID%' OR
             insurance_3 LIKE '%MEDICAID%' THEN 1
        WHEN insurance_1 LIKE '%MASSHEALTH%' OR
             insurance_2 LIKE '%MASSHEALTH%' OR
             insurance_3 LIKE '%MASSHEALTH%' THEN 1
        -- Add similar conditions for other government insurances as needed
        ELSE 0
    END;

-- Merge #tmp_demoInsurance into #A1c12MonthsLaterTable
MERGE INTO #A1c12MonthsLaterTable AS Target
USING #tmp_demoInsurance AS Source
ON Target.EMPI = Source.EMPI
WHEN MATCHED THEN 
    UPDATE SET Target.demoGovIns = Source.governmentInsurance;
--SELECT * FROM #A1c12MonthsLaterTable
--------------------------------------------------------------------------------------------------------
--Create dataset
--------------------------------------------------------------------------------------------------------
--initialize the DiabetesOutcomes table directly from #A1c12MonthsLaterTable
--include relevant columns from #tmp_studyPop and #tmp_PrimCarePatients

IF OBJECT_ID('dbo.DiabetesOutcomes', 'U') IS NOT NULL
    DROP TABLE dbo.DiabetesOutcomes;

-- Create the DiabetesOutcomes table with all columns from #A1c12MonthsLaterTable and additional columns
SELECT 
    a12.*, -- Select all columns from #A1c12MonthsLaterTable
    id.IndexDate AS ElevatedA1cDate, -- IndexDate as the date of the first elevated A1c measurement
    sp.AgeYears, -- AgeYears from #tmp_studyPop
    ppc.NumberEncounters -- NumberEncounters from #tmp_PrimCarePatients
INTO dbo.DiabetesOutcomes
FROM #A1c12MonthsLaterTable a12
INNER JOIN #tmp_indexDate id ON a12.EMPI = id.EMPI
INNER JOIN #tmp_studyPop sp ON a12.EMPI = sp.EMPI
INNER JOIN #tmp_PrimCarePatients ppc ON a12.EMPI = ppc.EMPI; --n=45711
--SELECT TOP 100 * FROM dbo.DiabetesOutcomes;

/*
-- Check cases where InitialA1cDate is earlier than ElevatedA1cDate
SELECT 
    EMPI,
    InitialA1cDate,
    ElevatedA1cDate,
    InitialA1c,
    A1cDateAfter12Months,
    A1cAfter12Months,
    A1cGreaterThan7,
    AgeYears,
    NumberEncounters
FROM dbo.DiabetesOutcomes
WHERE InitialA1cDate < ElevatedA1cDate;

-- Count how many such cases exist
SELECT COUNT(*)
FROM dbo.DiabetesOutcomes
WHERE InitialA1cDate < ElevatedA1cDate;
*/
-- Merge columns from selected RDPR tables into DiabetesOutcomes

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