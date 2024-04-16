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
) --n=65327

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

SELECT * FROM #A1c12MonthsLaterTable; --n=65327
--------------------------------------------------------------------------------------------------------
--Additional exclusion criteria
--------------------------------------------------------------------------------------------------------

--Exclude those whose age is < 18 years or > 75 years
IF OBJECT_ID('tempdb..#tmp_Under75') IS NOT NULL
    DROP TABLE #tmp_Under75;

SELECT 
    id.EMPI,
    id.IndexDate,
    DATEDIFF(YEAR, d.Date_of_Birth, id.IndexDate) AS AgeYears, -- age at IndexDate
    d.Gender_Legal_Sex,
    YEAR(d.Date_of_Birth) AS BirthYear -- Extracts the birth year
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
--Patient characteristics
--------------------------------------------------------------------------------------------------------

-- Add columns to #A1c12MonthsLaterTable if they don't exist
IF COL_LENGTH('#A1c12MonthsLaterTable', 'Gender') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD Gender varchar(50);
IF COL_LENGTH('#A1c12MonthsLaterTable', 'Female') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD Female int;
IF COL_LENGTH('#A1c12MonthsLaterTable', 'Marital') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD Marital varchar(50);
IF COL_LENGTH('#A1c12MonthsLaterTable', 'Married') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD Married int;
IF COL_LENGTH('#A1c12MonthsLaterTable', 'GovIns') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD GovIns int;
IF COL_LENGTH('#A1c12MonthsLaterTable', 'English') IS NULL
    ALTER TABLE #A1c12MonthsLaterTable ADD English int;

-- Update demographic data
UPDATE a
SET a.Gender = d.Gender_Legal_Sex,
    a.Female = CASE WHEN d.Gender_Legal_Sex = 'Female' THEN 1 ELSE 0 END,
    a.English = CASE WHEN d.Language LIKE 'ENG%' THEN 1 ELSE 0 END,
    a.Marital = CASE 
                        WHEN d.Marital_status IN ('Widow', 'Law', 'Partner', 'Married') THEN 'Married/Partnered'
                        WHEN d.Marital_status IN ('Separated', 'Divorce') THEN 'Separated'
                        WHEN d.Marital_status = 'Single' THEN 'Single'
                        ELSE 'Unknown'
                    END,
    a.Married = CASE WHEN d.Marital_status IN ('Married', 'Law', 'Partner') THEN 1 ELSE 0 END
FROM #A1c12MonthsLaterTable a
JOIN dem_2_pcp_combined d ON a.EMPI = d.EMPI; --n=65327

-- Update government insurance data
UPDATE a
SET a.GovIns = CASE 
                    WHEN c.insurance_1 LIKE '%MEDICARE%' OR c.insurance_2 LIKE '%MEDICARE%' OR c.insurance_3 LIKE '%MEDICARE%' THEN 1
                    WHEN c.insurance_1 LIKE '%MEDICAID%' OR c.insurance_2 LIKE '%MEDICAID%' OR c.insurance_3 LIKE '%MEDICAID%' THEN 1
                    WHEN c.insurance_1 LIKE '%MASSHEALTH%' OR c.insurance_2 LIKE '%MASSHEALTH%' OR c.insurance_3 LIKE '%MASSHEALTH%' THEN 1
                    ELSE 0
                END
FROM #A1c12MonthsLaterTable a
JOIN con_2_pcp_combined c ON a.EMPI = c.EMPI;

SELECT * FROM #A1c12MonthsLaterTable; --n=65327

--------------------------------------------------------------------------------------------------------
--SDI based on zip code
--------------------------------------------------------------------------------------------------------
-- Map SDI to zip code from contact table

IF OBJECT_ID('tempdb..#CleanedZipCodes') IS NOT NULL
    DROP TABLE #CleanedZipCodes;

-- Clean Zip codes
SELECT 
    EMPI, 
    CASE 
        -- Add leading zeros if less than 5 digits
        WHEN LEN(Zip) < 5 THEN RIGHT(CONCAT('00000', Zip), 5)
        -- Trim to first 5 digits if more than 5 digits
        ELSE LEFT(Zip, 5)
    END AS CleanedZip
INTO #CleanedZipCodes
FROM con_2_pcp_combined
WHERE EMPI IN (SELECT EMPI FROM #tmp_studyPop);
--SELECT * FROM #CleanedZipCodes; --n=64611

-- Map cleaned Zip codes to ZCTA5_FIPS and SDI score
--1. The #CleanedZipCodes table is joined with the ZIPCodetoZCTACrosswalk2022UDS using the cleaned zip codes.
--2. The ZIPCodetoZCTACrosswalk2022UDS is then joined with rgcsdi_2015_2019_zcta to bring in the SDI score, using the ZCTA codes matched from the crosswalk table.

IF OBJECT_ID('tempdb..#MappedZipToSDI') IS NOT NULL
    DROP TABLE #MappedZipToSDI;

SELECT 
    cz.EMPI, 
    cz.CleanedZip, 
    rz.SDI_score
INTO #MappedZipToSDI
FROM #CleanedZipCodes cz
INNER JOIN ZIPCodetoZCTACrosswalk2022UDS crosswalk ON cz.CleanedZip = crosswalk.ZIP_CODE
INNER JOIN dbo.rgcsdi_2015_2019_zcta rz ON crosswalk.zcta = rz.ZCTA5_FIPS;

--SELECT * FROM #MappedZipToSDI; --n=62481

--------------------------------------------------------------------------------------------------------
--Additional variables from Demographics table
--------------------------------------------------------------------------------------------------------

IF OBJECT_ID('tempdb..#VeteranStatus') IS NOT NULL
    DROP TABLE #VeteranStatus;

SELECT 
    d.EMPI,
    CASE WHEN d.Is_a_veteran = 'Yes' THEN 1 ELSE 0 END AS Veteran
INTO #VeteranStatus
FROM dem_2_pcp_combined d
INNER JOIN #tmp_studyPop sp ON d.EMPI = sp.EMPI; --n 64611
--SELECT * FROM #VeteranStatus;

--------------------------------------------------------------------------------------------------------
--Create dataset
--------------------------------------------------------------------------------------------------------

--initialize the DiabetesOutcomes table directly from #A1c12MonthsLaterTable
--include relevant columns from #tmp_studyPop, #tmp_PrimCarePatients
IF OBJECT_ID('dbo.DiabetesOutcomes', 'U') IS NOT NULL
    DROP TABLE dbo.DiabetesOutcomes;

-- Create the DiabetesOutcomes table with a new study-specific ID
-- Keep EMPI for now
SELECT 
    ROW_NUMBER() OVER (ORDER BY a12.EMPI) AS studyID, -- Generate a unique identifier
    a12.EMPI,
    a12.IndexDate,
    a12.InitialA1c,
    a12.A1cGreaterThan7,
    a12.Female,
    a12.Married,
    a12.GovIns,
    a12.English,
    DATEDIFF(DAY, id.IndexDate, a12.InitialA1cDate) AS DaysFromIndexToInitialA1cDate,
    DATEDIFF(DAY, id.IndexDate, a12.A1cDateAfter12Months) AS DaysFromIndexToA1cDateAfter12Months,
    DATEDIFF(DAY, id.IndexDate, id.FirstEncounterDate) AS DaysFromIndexToFirstEncounterDate,
    DATEDIFF(DAY, id.IndexDate, id.LastEncounterDate) AS DaysFromIndexToLastEncounterDate,
    DATEDIFF(DAY, id.IndexDate, id.LatestDate) AS DaysFromIndexToLatestDate,
    DATEDIFF(DAY, id.IndexDate, id.PatientTurns18) AS DaysFromIndexToPatientTurns18,
   -- id.IndexDate, -- Keeping the original IndexDate for reference
    sp.AgeYears -- AgeYears from #tmp_studyPop
INTO dbo.DiabetesOutcomes
FROM #A1c12MonthsLaterTable a12
INNER JOIN #tmp_indexDate id ON a12.EMPI = id.EMPI
INNER JOIN #tmp_studyPop sp ON a12.EMPI = sp.EMPI
INNER JOIN #tmp_PrimCarePatients ppc ON a12.EMPI = ppc.EMPI;--n=55667
SELECT * FROM dbo.DiabetesOutcomes;

-- Add the SDI_score column to DiabetesOutcomes if it doesn't exist
IF COL_LENGTH('dbo.DiabetesOutcomes', 'SDI_score') IS NULL
    ALTER TABLE dbo.DiabetesOutcomes ADD SDI_score FLOAT;

-- Update DiabetesOutcomes with SDI_score from #MappedZipToSDI
UPDATE do
SET do.SDI_score = mzd.SDI_score
FROM dbo.DiabetesOutcomes do
INNER JOIN #MappedZipToSDI mzd ON do.EMPI = mzd.EMPI; --n=53833 (1834 missing)
--SELECT * FROM dbo.DiabetesOutcomes;

-- Add the Veteran column to DiabetesOutcomes if it doesn't exist
IF COL_LENGTH('dbo.DiabetesOutcomes', 'Veteran') IS NULL
    ALTER TABLE dbo.DiabetesOutcomes ADD Veteran FLOAT;

UPDATE do
SET do.Veteran = vs.Veteran
FROM dbo.DiabetesOutcomes do
INNER JOIN #VeteranStatus vs ON do.EMPI = vs.EMPI;
SELECT * FROM dbo.DiabetesOutcomes; --n=55667

--------------------------------------------------------------------------------------------------------
--Preprocess diagnoses table (export to file)
--------------------------------------------------------------------------------------------------------

IF OBJECT_ID('dbo.Diagnoses', 'U') IS NOT NULL DROP TABLE dbo.Diagnoses;

;WITH CTE_Diagnoses AS (
    SELECT
        dia.EMPI,
        dia.Date,
        CASE 
            WHEN dia.Code_Type IN ('ICD9', 'ICD10') AND CHARINDEX('.', dia.Code) > 0 
            THEN LEFT(dia.Code, CHARINDEX('.', dia.Code)) + 
                 SUBSTRING(dia.Code, CHARINDEX('.', dia.Code) + 1, 1) -- Keep only first digit after decimal
            ELSE dia.Code 
        END AS Code,
        dia.Code_Type,
        do.IndexDate,
        CASE WHEN dia.Date <= do.IndexDate THEN 1 ELSE 0 END AS DiagnosisBeforeOrOnIndexDate,
        CASE 
            WHEN CHARINDEX('.', dia.Code) > 0 
            THEN LEFT(dia.Code, CHARINDEX('.', dia.Code) - 1) + 
                 '.' + 
                 SUBSTRING(dia.Code, CHARINDEX('.', dia.Code) + 1, 1) 
            ELSE dia.Code 
        END + 
        '_' + dia.Code_Type AS CodeWithType
    FROM dia_2_pcp_combined dia
    INNER JOIN DiabetesOutcomes do ON dia.EMPI = do.EMPI
    WHERE dia.Code_Type IN ('ICD9', 'ICD10') AND dia.Code IS NOT NULL AND dia.Code <> ''
),
AggregatedDiagnoses AS (
    SELECT
        EMPI,
        Code,
        Code_Type,
        MAX(Date) AS LatestDate,
        MAX(IndexDate) AS LatestIndexDate,
        CodeWithType
    FROM CTE_Diagnoses
    WHERE DiagnosisBeforeOrOnIndexDate = 1
    GROUP BY EMPI, Code, Code_Type, CodeWithType
)

SELECT 
    EMPI,
    LatestDate AS Date,
    Code,
    Code_Type,
    LatestIndexDate AS IndexDate,
    CodeWithType
INTO dbo.Diagnoses
FROM AggregatedDiagnoses;

SELECT TOP 100 * FROM dbo.Diagnoses;

-- Count distinct values of CodeWithType and EMPI in Diagnoses table
SELECT 
    COUNT(DISTINCT EMPI) AS UniqueEMPIs,
    COUNT(DISTINCT CodeWithType) AS UniqueCodes
FROM dbo.Diagnoses;

--------------------------------------------------------------------------------------------------------
--Preprocess procedures table (export to file)
--------------------------------------------------------------------------------------------------------

IF OBJECT_ID('dbo.Procedures', 'U') IS NOT NULL DROP TABLE dbo.Procedures;

;WITH CTE_Procedures AS (
    SELECT
        prc.EMPI,
        prc.Date,
        CASE 
            WHEN prc.Code_Type IN ('ICD9', 'ICD10') AND CHARINDEX('.', prc.Code) > 0 
            THEN LEFT(prc.Code, CHARINDEX('.', prc.Code)) + 
                 SUBSTRING(prc.Code, CHARINDEX('.', prc.Code) + 1, 1) -- Keep only first digit after decimal
            ELSE prc.Code 
        END AS Code,
        prc.Code_Type,
        po.IndexDate,
        CASE WHEN prc.Date <= po.IndexDate THEN 1 ELSE 0 END AS ProcedureBeforeOrOnIndexDate,
        CASE 
            WHEN CHARINDEX('.', prc.Code) > 0 
            THEN LEFT(prc.Code, CHARINDEX('.', prc.Code) - 1) + 
                 '.' + 
                 SUBSTRING(prc.Code, CHARINDEX('.', prc.Code) + 1, 1) 
            ELSE prc.Code 
        END + 
        '_' + prc.Code_Type AS CodeWithType
    FROM prc_2_pcp_combined prc
    INNER JOIN DiabetesOutcomes po ON prc.EMPI = po.EMPI
    WHERE prc.Code_Type IN ('CPT', 'ICD9', 'ICD10') AND prc.Code IS NOT NULL AND prc.Code <> ''
),
AggregatedProcedures AS (
    SELECT
        EMPI,
        Code,
        Code_Type,
        MAX(Date) AS LatestDate,
        MAX(IndexDate) AS LatestIndexDate,
        CodeWithType
    FROM CTE_Procedures
    WHERE ProcedureBeforeOrOnIndexDate = 1
    GROUP BY EMPI, Code, Code_Type, CodeWithType
)

SELECT 
    EMPI,
    LatestDate AS Date,
    Code,
    Code_Type,
    LatestIndexDate AS IndexDate,
    CodeWithType
INTO dbo.Procedures
FROM AggregatedProcedures;

SELECT TOP 100 * FROM dbo.Procedures;

-- Count distinct values of CodeWithType and EMPI in Procedures table
SELECT 
    COUNT(DISTINCT EMPI) AS UniqueEMPIs,
    COUNT(DISTINCT CodeWithType) AS UniqueCodes
FROM dbo.Procedures;

--UniqueEMPIs   UniqueCodes
-- 53675    10068

--------------------------------------------------------------------------------------------------------
--Preprocess Labs table (export to file)
--------------------------------------------------------------------------------------------------------

IF OBJECT_ID('dbo.Labs', 'U') IS NOT NULL DROP TABLE dbo.Labs;

;WITH CombinedLabs AS (
    SELECT
        e.empi,
        e.LabDate AS Date,
        e.StudyLabCode AS Code,
        e.NVal AS Result,
        e.ValType,
        'EPIC' AS Source
    FROM dbo.labsEPIC e
    WHERE e.LabDate BETWEEN @minDate AND @maxDate
      AND e.NVal IS NOT NULL
    UNION ALL
    SELECT
        l.empi,
        l.LabDate,
        l.GroupCD AS Code,
        CAST(l.NVal AS FLOAT) AS Result,
        l.ValType,
        'LMR' AS Source
    FROM dbo.labsLMR l
    WHERE l.LabDate BETWEEN @minDate AND @maxDate
      AND l.NVal IS NOT NULL
    UNION ALL
    SELECT
        a.EMPI,
        a.LabDate,
        a.GroupCD AS Code,
        CAST(a.nval AS FLOAT) AS Result,
        a.valtype,
        'Archive' AS Source
    FROM dbo.labsLMRArchive a
    WHERE a.LabDate BETWEEN @minDate AND @maxDate
      AND a.nval IS NOT NULL
)
SELECT
    cl.EMPI,
    cl.Date,
    cl.Code,
    cl.Result,
    cl.ValType,
    cl.Source
INTO dbo.Labs
FROM CombinedLabs cl
INNER JOIN #tmp_indexDate idx ON cl.EMPI = idx.EMPI AND cl.Date <= idx.IndexDate
WHERE cl.Result IS NOT NULL;

SELECT TOP 100 * FROM dbo.Labs;

-- Further diagnostic queries to inspect the result set
SELECT 
    COUNT(DISTINCT EMPI) AS UniqueEMPIs,
    COUNT(DISTINCT Code) AS UniqueLabCodes
FROM dbo.Labs;

--UniqueEMPIs   UniqueLabCodes
--74960 3145