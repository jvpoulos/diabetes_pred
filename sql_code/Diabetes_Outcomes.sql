--Modified from Q:\Projects\DiabetesPredictiveModeling\SQL_examples\DataProcessingExamples\SampleCodeFromDiabetesOutcomesProjectV05.sql
              --Q:\Projects\DiabetesPredictiveModeling\SQL_examples\DataProcessingExamples\SnippetsforInclusionCriteria.sql

--creates Final analysis table: DiabetesOutcomes..DiabetesOutcomesAnalysisDataset

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

--DECLARE @FollowUpMonths INT = 12;

SET @min_A1c = 4.0
SET @max_A1c = 20.0
DECLARE @min_A1c_inclusion FLOAT = 7.0

--start with n=91557
--------------------------------------------------------------------------------------------------------
--Inclusion and exlusion criteria
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

--HbA1c ≥ 7.0% during the study period

IF OBJECT_ID('tempdb..#tmp_A1cElevated') IS NOT NULL
    DROP TABLE #tmp_A1cElevated;

-- Create a temporary table to store dates of elevated A1c along with additional columns
CREATE TABLE #tmp_A1cElevated (
    EMPI VARCHAR(50),
    A1cDate DATE,
    Age INT,
    Gender VARCHAR(10),
    LabDate DATE,
    GroupCD VARCHAR(255),
    nval FLOAT
)

-- Insert into #tmp_A1cElevated from labsLMR
INSERT INTO #tmp_A1cElevated (EMPI, A1cDate, Age, Gender, LabDate, GroupCD, nval)
SELECT DISTINCT 
    b.EMPI, 
    b.LabDate AS A1cDate, 
    DATEDIFF(year, p.Date_of_Birth, b.LabDate) AS Age, 
    p.Gender_Legal_Sex AS Gender,
    b.LabDate,
    b.GroupCD,
    TRY_CONVERT(FLOAT, b.nval) AS nval
FROM dbo.labsLMR b
JOIN dem_2_pcp_combined p ON p.EMPI = b.EMPI
WHERE b.LabDate > @minEntryDate 
  AND b.GroupCD LIKE '%A1c%'
  AND TRY_CONVERT(FLOAT, b.nval) >= @min_A1c_inclusion; --n=375674

-- Insert into #tmp_A1cElevated from labsLMRArchive
INSERT INTO #tmp_A1cElevated (EMPI, A1cDate, Age, Gender, LabDate, GroupCD, nval)
SELECT DISTINCT 
    a.EMPI, 
    a.LabDate AS A1cDate, 
    DATEDIFF(year, p.Date_of_Birth, a.LabDate) AS Age, 
    p.Gender_Legal_Sex AS Gender,
    a.LabDate,
    a.GroupCD,
    TRY_CONVERT(FLOAT, a.nval) AS nval
FROM dbo.labsLMRArchive a
JOIN dem_2_pcp_combined p ON p.EMPI = a.EMPI
WHERE a.LabDate > @minEntryDate 
  AND a.GroupCD LIKE '%A1c%'
  AND TRY_CONVERT(FLOAT, a.nval) >= @min_A1c_inclusion; --n=205692

-- Insert into #tmp_A1cElevated from labsEPIC
INSERT INTO #tmp_A1cElevated (EMPI, A1cDate, Age, Gender, LabDate, GroupCD, nval)
SELECT DISTINCT 
    e.EMPI, 
    e.LabDate AS A1cDate, 
    DATEDIFF(year, p.Date_of_Birth, e.LabDate) AS Age, 
    p.Gender_Legal_Sex AS Gender,
    e.LabDate,
    e.StudyLabCode AS GroupCD,
    TRY_CONVERT(FLOAT, e.nval) AS nval
FROM labsEPIC e
JOIN dem_2_pcp_combined p ON p.EMPI = e.EMPI
WHERE e.LabDate > @minEntryDate 
  AND e.StudyLabCode LIKE '%A1c%'
  AND TRY_CONVERT(FLOAT, e.nval) >= @min_A1c_inclusion; --n=697535

IF OBJECT_ID('tempdb..#tmp_indexDate') IS NOT NULL
    DROP TABLE #tmp_indexDate;

-- Determine Eligible Periods and Index Date
SELECT p.EMPI, MIN(a.A1cDate) AS IndexDate
INTO #tmp_indexDate
FROM #tmp_PrimCarePatients p
INNER JOIN #tmp_A1cElevated a ON a.EMPI = p.EMPI
INNER JOIN dem_2_pcp_combined d ON p.EMPI = d.EMPI
WHERE a.A1cDate BETWEEN p.FirstEncounterDate AND p.LastEncounterDate
      AND TRY_CONVERT(FLOAT, a.nval) >= @min_A1c_inclusion
      AND DATEDIFF(year, d.Date_of_Birth, a.A1cDate) BETWEEN 18 AND 75
   --   AND (d.Date_of_Death IS NULL OR DATEDIFF(month, a.A1cDate, TRY_CONVERT(DATE, d.Date_of_Death, 101)) > @FollowUpMonths)
      AND a.A1cDate BETWEEN @minEntryDate AND @maxEntryDate
GROUP BY p.EMPI; --n=61103

--------------------------------------------------------------------------------------------------------
--Additional exclusion criteria
--------------------------------------------------------------------------------------------------------

--Exclude those whose age is < 18 years or > 75 years (done in prev. step)
IF OBJECT_ID('tempdb..#tmp_Under75') IS NOT NULL
    DROP TABLE #tmp_Under75

SELECT a.*
INTO #tmp_Under75
FROM 
(
    SELECT i.*, DATEDIFF(year, p.Date_of_Birth, i.IndexDate) AS AgeYears, p.Gender, p.Date_of_Birth
    FROM #tmp_indexDate i
    JOIN #tmp_PrimCarePatients p ON p.EMPI = i.EMPI
) AS a
WHERE a.AgeYears BETWEEN 18 AND 75 --n=61103

IF OBJECT_ID('tempdb..#tmp_studyPop') IS NOT NULL
    DROP TABLE #tmp_studyPop

--Exclusion criteria: exclude those with unknown gender
select *
into #tmp_studyPop
from #tmp_Under75
where gender in ('Female', 'Male') --n=61100

--------------------------------------------------------------------------------------------------------
--A1c data
--------------------------------------------------------------------------------------------------------

--Clean A1c Values for those in study pop.
IF OBJECT_ID('tempdb..#CleanedA1cAverages') IS NOT NULL
    DROP TABLE #CleanedA1cAverages;

SELECT e.EMPI, e.A1cDate, AVG(e.A1c) AS A1c
INTO #CleanedA1cAverages
FROM (
    SELECT l.EMPI, CAST(l.LabDate AS DATE) AS A1cDate, l.nval AS 'A1c'
    FROM labsEPIC l
    INNER JOIN #tmp_studyPop s ON l.EMPI = s.EMPI
    WHERE l.StudyLabCode LIKE '%A1c%' 
          AND ISNUMERIC(l.nval) > 0 
          AND l.nval >= @min_A1c AND l.nval <= @max_A1c
    UNION ALL
    SELECT lm.EMPI, CAST(lm.LabDate AS DATE) AS A1cDate, lm.nval AS 'A1c'
    FROM labsLMR lm
    INNER JOIN #tmp_studyPop s ON lm.EMPI = s.EMPI
    WHERE lm.GroupCD LIKE '%A1c%'
          AND ISNUMERIC(lm.nval) > 0 
          AND lm.nval >= @min_A1c AND lm.nval <= @max_A1c
    UNION ALL
    SELECT la.EMPI, CAST(la.LabDate AS DATE) AS A1cDate, la.nval AS 'A1c'
    FROM labsLMRArchive la
    INNER JOIN #tmp_studyPop s ON la.EMPI = s.EMPI
    WHERE la.GroupCD LIKE '%A1c%' 
          AND ISNUMERIC(la.nval) > 0 
          AND la.nval >= @min_A1c AND la.nval <= @max_A1c
) AS e
GROUP BY e.EMPI, e.A1cDate; --n=1435435

--------------------------------------------------------------------------------------------------------
--Identify hyperglycemic periods
--------------------------------------------------------------------------------------------------------

--1.Identify the first elevated A1c measurement (7 or greater) for each patient.
--2.Find the A1c measurement approximately 12 months later.
--3.Handle cases where the A1c fluctuates during the year.

IF OBJECT_ID('dbo.A1c12MonthsLaterTable', 'U') IS NOT NULL
    DROP TABLE dbo.A1c12MonthsLaterTable;

-- Step 1: Identify the first elevated A1c measurement for each patient in #tmp_studyPop
WITH FirstElevatedA1c AS (
    SELECT 
        c.EMPI, 
        c.A1cDate,
        c.A1c,
        ROW_NUMBER() OVER (PARTITION BY c.EMPI ORDER BY c.A1cDate) AS RowNum
    FROM #CleanedA1cAverages c
    INNER JOIN #tmp_studyPop s ON c.EMPI = s.EMPI
    WHERE c.A1c >= 7
),
-- Step 2: Find all A1c measurements approximately 12 months later for those in #tmp_studyPop
AllA1c12MonthsLater AS (
    SELECT 
        f.EMPI,
        f.A1cDate AS InitialA1cDate,
        f.A1c AS InitialA1c,
        a.A1c AS A1cAfter12Months,
        a.A1cDate AS A1cDateAfter12Months,
        CASE 
            WHEN a.A1c >= 7 THEN 1
            ELSE 0
        END AS A1cGreaterThan7,
        ROW_NUMBER() OVER (
            PARTITION BY f.EMPI 
            ORDER BY ABS(DATEDIFF(day, DATEADD(month, 12, f.A1cDate), a.A1cDate))
        ) AS RowNum
    FROM FirstElevatedA1c f
    LEFT JOIN #CleanedA1cAverages a ON f.EMPI = a.EMPI
        AND a.A1cDate >= DATEADD(month, 9, f.A1cDate)
        AND a.A1cDate <= DATEADD(month, 15, f.A1cDate)
    WHERE f.RowNum = 1
)

-- Final selection
SELECT
    EMPI,
    InitialA1cDate,
    InitialA1c,
    A1cDateAfter12Months,
    A1cAfter12Months,
    A1cGreaterThan7
INTO A1c12MonthsLaterTable -- Save as a table
FROM AllA1c12MonthsLater
WHERE RowNum = 1 AND A1cAfter12Months IS NOT NULL; --n=40363 -- Ensures that only the closest A1c measurement within the timeframe is selected

--------------------------------------------------------------------------------------------------------
--Create dataset
--------------------------------------------------------------------------------------------------------

-- uses the A1c measurement from the A1c12MonthsLaterTable for generating the 12-month follow-up in the DiabetesOutcomes table
IF OBJECT_ID('dbo.DiabetesOutcomes', 'U') IS NOT NULL
    DROP TABLE dbo.DiabetesOutcomes;

WITH UniqueIndexDate AS (
    SELECT 
        EMPI,
        MAX(IndexDate) AS LatestIndexDate -- or use MIN
    FROM #tmp_indexDate
    GROUP BY EMPI
)
SELECT 
    s.EMPI,
    a12.InitialA1cDate AS ElevatedA1cDate,
    id.LatestIndexDate AS IndexDate,
    a12.A1cDateAfter12Months,
    a12.A1cAfter12Months
INTO DiabetesOutcomes
FROM #tmp_studyPop s
LEFT JOIN UniqueIndexDate id ON s.EMPI = id.EMPI
INNER JOIN A1c12MonthsLaterTable a12 ON s.EMPI = a12.EMPI;--n=40363

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
    a12.A1cAfter12Months, --n=40363

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
FROM A1c12MonthsLaterTable a12
LEFT JOIN con_2_pcp_combined conp ON a12.EMPI = conp.EMPI
LEFT JOIN dem_2_pcp_combined demp ON a12.EMPI = demp.EMPI
LEFT JOIN (
    SELECT 
        EMPI,
        MAX(IndexDate) AS LatestIndexDate
    FROM #tmp_indexDate
    GROUP BY EMPI
) id ON a12.EMPI = id.EMPI; --n=40363
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