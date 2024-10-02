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
    WHERE l.ComponentCommonNM LIKE '%A1c%' 
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
GROUP BY e.EMPI, e.A1cDate; --n=1958797
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
  AND ca.A1cDate > @minEntryDate; --n=997739
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
CREATE INDEX idx_a1c ON #A1cMeasurements(EMPI, A1cDate); --n=889223
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
WHERE ad.A1cValue IS NOT NULL --n=76455

-- Delete 431 rows where IndexDate exceeds @maxEntryDate or LastEncounterDate
DELETE FROM #tmp_indexDate
WHERE 
    (IndexDate > @maxEntryDate) OR 
    (IndexDate > LastEncounterDate)

SELECT * FROM #tmp_indexDate --n =76029

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
) --n=65315

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

SELECT * FROM #A1c12MonthsLaterTable; --n=65315
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
WHERE DATEDIFF(YEAR, d.Date_of_Birth, id.IndexDate) BETWEEN 18 AND 75; --n=65315

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
JOIN dem_2_pcp_combined d ON a.EMPI = d.EMPI; --n=64608

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

SELECT * FROM #A1c12MonthsLaterTable; --n=65315

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
--SELECT * FROM #CleanedZipCodes; --n=65315

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

--SELECT * FROM #MappedZipToSDI; --n=65315

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
INNER JOIN #tmp_studyPop sp ON d.EMPI = sp.EMPI; --n=64608
--SELECT * FROM #VeteranStatus;

--------------------------------------------------------------------------------------------------------
--Create dataset
--------------------------------------------------------------------------------------------------------

--initialize the DiabetesOutcomes table directly from #A1c12MonthsLaterTable
--include relevant columns from #tmp_studyPop, #tmp_PrimCarePatients
IF OBJECT_ID('dbo.DiabetesOutcomes_old', 'U') IS NOT NULL
    DROP TABLE dbo.DiabetesOutcomes_old;

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
INTO dbo.DiabetesOutcomes_old
FROM #A1c12MonthsLaterTable a12
INNER JOIN #tmp_indexDate id ON a12.EMPI = id.EMPI
INNER JOIN #tmp_studyPop sp ON a12.EMPI = sp.EMPI
INNER JOIN #tmp_PrimCarePatients ppc ON a12.EMPI = ppc.EMPI;--n=55667
SELECT * FROM dbo.DiabetesOutcomes_old;

-- Add the SDI_score column to DiabetesOutcomes if it doesn't exist
IF COL_LENGTH('dbo.DiabetesOutcomes', 'SDI_score') IS NULL
    ALTER TABLE dbo.DiabetesOutcomes_old ADD SDI_score FLOAT;

-- Update DiabetesOutcomes with SDI_score from #MappedZipToSDI
UPDATE do
SET do.SDI_score = mzd.SDI_score
FROM dbo.DiabetesOutcomes_old do
INNER JOIN #MappedZipToSDI mzd ON do.EMPI = mzd.EMPI; --n=53833 (1834 missing)
--SELECT * FROM dbo.DiabetesOutcomes;

-- Add the Veteran column to DiabetesOutcomes if it doesn't exist
IF COL_LENGTH('dbo.DiabetesOutcomes_old', 'Veteran') IS NULL
    ALTER TABLE dbo.DiabetesOutcomes_old ADD Veteran FLOAT;

UPDATE do
SET do.Veteran = vs.Veteran
FROM dbo.DiabetesOutcomes_old do
INNER JOIN #VeteranStatus vs ON do.EMPI = vs.EMPI;
SELECT * FROM dbo.DiabetesOutcomes_old; --n=55667

--------------------------------------------------------------------------------------------------------
--Preprocess diagnoses table (export to file)
--------------------------------------------------------------------------------------------------------

-- original old Diagnoses 

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
FROM dbo.Diagnoses_old;
--54154 Unique EMPIs, 12684 Unique Codes

--------------------------------------------------------------------------------------------------------
--Preprocess procedures table (export to file)
--------------------------------------------------------------------------------------------------------
--original old Procedures

--IF OBJECT_ID('dbo.Procedures_old', 'U') IS NOT NULL DROP TABLE dbo.Procedures_old;

--;WITH CTE_Procedures AS (
--    SELECT
--        prc.EMPI,
--        prc.Date,
--        CASE 
--            WHEN prc.Code_Type IN ('ICD9', 'ICD10') AND CHARINDEX('.', prc.Code) > 0 
--            THEN LEFT(prc.Code, CHARINDEX('.', prc.Code)) + 
--                 SUBSTRING(prc.Code, CHARINDEX('.', prc.Code) + 1, 1) -- Keep only first digit after decimal
--            ELSE prc.Code 
--        END AS Code,
--        prc.Code_Type,
--        po.IndexDate,
--        CASE WHEN prc.Date <= po.IndexDate THEN 1 ELSE 0 END AS ProcedureBeforeOrOnIndexDate,
--        CASE 
--            WHEN CHARINDEX('.', prc.Code) > 0 
--            THEN LEFT(prc.Code, CHARINDEX('.', prc.Code) - 1) + 
--                 '.' + 
--                 SUBSTRING(prc.Code, CHARINDEX('.', prc.Code) + 1, 1) 
--            ELSE prc.Code 
--        END + 
--        '_' + prc.Code_Type AS CodeWithType
--    FROM prc_2_pcp_combined prc
--    INNER JOIN DiabetesOutcomes_old po ON prc.EMPI = po.EMPI
--    WHERE prc.Code_Type IN ('CPT', 'ICD9', 'ICD10') AND prc.Code IS NOT NULL AND prc.Code <> ''
--),
--AggregatedProcedures AS (
--    SELECT
--        EMPI,
--        Code,
--        Code_Type,
--        MAX(Date) AS LatestDate,
--        MAX(IndexDate) AS LatestIndexDate,
--        CodeWithType
--    FROM CTE_Procedures
--    WHERE ProcedureBeforeOrOnIndexDate = 1
--    GROUP BY EMPI, Code, Code_Type, CodeWithType
--)
--SELECT 
--    EMPI,
--    LatestDate AS Date,
--    Code,
--    Code_Type,
--    LatestIndexDate AS IndexDate,
--    CodeWithType
--INTO dbo.Procedures_old
--FROM AggregatedProcedures;

--SELECT TOP 100 * FROM dbo.Procedures_old;



-- Create a temporary table to store the initial processing
IF OBJECT_ID('tempdb..#CTE_Procedures', 'U') IS NOT NULL DROP TABLE #CTE_Procedures;

;;WITH CTE_Procedures AS (
    SELECT
        s.StudyID,
        prc.Date,
        prc.Code,
        prc.Code_Type,
        po.IndexDate,
        CASE WHEN prc.Date <= po.IndexDate THEN 1 ELSE 0 END AS ProcedureBeforeOrOnIndexDate,
        prc.Code + '_' + prc.Code_Type AS CodeWithType
    FROM prc_2_pcp_combined prc
    INNER JOIN StudyID s ON prc.EMPI = s.EMPI
    INNER JOIN DiabetesOutcomes po ON s.StudyID = po.StudyID
    WHERE prc.Code_Type IN ('CPT', 'ICD9', 'ICD10') AND prc.Code IS NOT NULL AND prc.Code <> ''
),
AggregatedProcedures AS (
    SELECT
        StudyID,
        Code,
        Code_Type,
        MAX(Date) AS LatestDate,
        MAX(IndexDate) AS LatestIndexDate,
        CodeWithType
    FROM CTE_Procedures
    WHERE ProcedureBeforeOrOnIndexDate = 1
    GROUP BY StudyID, Code, Code_Type, CodeWithType
)

SELECT 
    StudyID,
    LatestDate AS Date,
    Code,
    Code_Type,
    LatestIndexDate AS IndexDate,
    CodeWithType
INTO #CTE_Procedures
FROM AggregatedProcedures;

SELECT COUNT(DISTINCT Code) FROM Procedures_ICD10
WHERE Code_Type = 'ICD9';



SELECT TOP 100 * FROM #CTE_Procedures;

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
--declarations deleted 

IF OBJECT_ID('dbo.Labs_old', 'U') IS NOT NULL DROP TABLE dbo.Labs_old;

;WITH CombinedLabs AS (
    SELECT
        e.EMPI,
        e.LabDate AS Date,
        e.ComponentCommonNM AS Code,
        TRY_CAST(e.NVal AS FLOAT) AS Result_n, -- Safe conversion to float
        e.TVal AS Result_t, -- Keep character results directly
        CASE WHEN TRY_CAST(e.NVal AS FLOAT) IS NOT NULL THEN 'Numeric' ELSE 'String' END AS ValType, -- Determine ValType
        'EPIC' AS Source
    FROM dbo.labsEpic e
    WHERE e.LabDate BETWEEN @minDate AND @maxDate
          AND (e.NVal IS NOT NULL OR e.TVal IS NOT NULL)
    UNION ALL
    SELECT
        l.empi,
        l.LabDate,
        l.GroupCD AS Code,
        TRY_CAST(l.NVal AS FLOAT) AS Result_n,
        l.TVal AS Result_t,
        CASE WHEN TRY_CAST(l.NVal AS FLOAT) IS NOT NULL THEN 'Numeric' ELSE 'String' END AS ValType,
        'LMR' AS Source
    FROM dbo.labsLMR l
    WHERE l.LabDate BETWEEN @minDate AND @maxDate
          AND (l.NVal IS NOT NULL OR l.TVal IS NOT NULL)
    UNION ALL
    SELECT
        a.EMPI,
        a.LabDate,
        a.GroupCD AS Code,
        TRY_CAST(a.nval AS FLOAT) AS Result_n,
        a.tval AS Result_t,
        CASE WHEN TRY_CAST(a.nval AS FLOAT) IS NOT NULL THEN 'Numeric' ELSE 'String' END AS ValType,
        'Archive' AS Source
    FROM dbo.labsLMRArchive a
    WHERE a.LabDate BETWEEN @minDate AND @maxDate
          AND (a.nval IS NOT NULL OR a.tval IS NOT NULL)
),
LabsWithIndexDate AS (
    SELECT
        cl.EMPI,
        cl.Date,
        cl.Code,
        cl.Result_n,
        cl.Result_t,
        cl.ValType,
        cl.Source
    FROM CombinedLabs cl
    INNER JOIN #tmp_indexDate idx ON cl.EMPI = idx.EMPI AND cl.Date <= idx.IndexDate
),
LabsFiltered AS (
    SELECT
        l.EMPI,
        l.Date,
        l.Code,
        l.Result_n,
        CASE WHEN l.Result_n IS NULL THEN l.Result_t ELSE NULL END AS Result_t, -- Keep only one result per row
        l.ValType,
        l.Source
    FROM LabsWithIndexDate l
    WHERE l.EMPI IN (SELECT EMPI FROM DiabetesOutcomes) -- Restrict to EMPIs present in DiabetesOutcomes
)

SELECT
    EMPI,
    Date,
    Code,
    Result_n,
    Result_t,
    ValType,
    Source
INTO dbo.Labs_old
FROM LabsFiltered;

SELECT TOP 100 * FROM dbo.Labs_old;

-- Further diagnostic queries to inspect the result set
SELECT 
    COUNT(DISTINCT EMPI) AS UniqueEMPIs,
    COUNT(DISTINCT Code) AS UniqueLabCodes
FROM dbo.Labs;

--UniqueEMPIs   UniqueLabCodes
--54775 11430

--Combining Labs Datasets 

-- delete temporary tables

IF OBJECT_ID('tempdb.dbo.#tmp_componentCommonNM') IS NOT NULL DROP TABLE #tmp_componentCommonNM

IF OBJECT_ID('tempdb.dbo.#tmp_groupNM') IS NOT NULL DROP TABLE #tmp_groupNM

IF OBJECT_ID('tempdb.dbo.#tmp_ExternalNM') IS NOT NULL DROP TABLE #tmp_ExternalNM

IF OBJECT_ID('tempdb.dbo.#tmp_weights') IS NOT NULL DROP TABLE #tmp_weights

IF OBJECT_ID('tempdb.dbo.#tmp_weights') IS NOT NULL DROP TABLE #tmp_weights

-- Step 1: Create a view for unique entries for ComponentCommonNM in DM2023..labsEpic
CREATE VIEW vw_tmp_componentCommonNM AS
SELECT le.ComponentCommonNM, COUNT(*) AS 'Num'
--INTO #tmp_componentCommonNM
FROM DM2023..labsEpic le
GROUP BY le.ComponentCommonNM;

SELECT TOP(1000) * 
FROM vw_tmp_componentCommonNM;



-- Step 2: Create a view for unique entries for GroupNM in DM2023..labsLMR
CREATE VIEW vw_tmp_groupNM AS
SELECT llmr.GroupNM, COUNT(*) AS 'Num'
FROM DM2023..labsLMR llmr
GROUP BY llmr.GroupNM;

SELECT *
FROM vw_tmp_groupNM;

-- Step 3: Create a view for matching entries between GroupNM and ComponentCommonNM
CREATE VIEW vw_matching_labs AS
SELECT gnm.GroupNM, ccn.ComponentCommonNM, gnm.Num
FROM vw_tmp_groupNM gnm
JOIN vw_tmp_componentCommonNM ccn ON gnm.GroupNM = ccn.ComponentCommonNM


SELECT TOP(1000) * 
FROM vw_matching_labs
ORDER BY Num DESC;


-- Step 4: Create views to use only matching labs for further analysis
CREATE VIEW vw_filtered_labsLMR AS
SELECT llmr.*
FROM DM2023..labsLMR llmr
JOIN vw_matching_labs ml ON llmr.GroupNM = ml.GroupNM;

SELECT *
FROM vw_filtered_labsLMR;

CREATE VIEW vw_filtered_labsEpic AS
SELECT le.*
FROM DM2023..labsEpic le
JOIN vw_matching_labs ml ON le.ComponentCommonNM = ml.ComponentCommonNM;

SELECT * 
FROM vw_filtered_labsEpic;

-- Step 5: Create a view to ensure labs in DM2023..labsLMRarchive have matching GroupNM values as in DM2023..labsLMR
CREATE VIEW vw_filtered_labsLMRarchive AS
SELECT la.*
FROM DM2023..labsLMRarchive la
JOIN vw_filtered_labsLMR fl ON la.GroupNM = fl.GroupNM;

SELECT *
FROM vw_filtered_labsLMRarchive;


--combine attempt
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


IF OBJECT_ID('dbo.Labs_old_combined', 'U') IS NOT NULL DROP TABLE dbo.Labs_old_combined;

;WITH CombinedLabs AS (
    SELECT
        e.EMPI,
        e.LabDate AS Date,
        e.ComponentCommonNM AS Code,
        COALESCE(CAST(e.NVal AS VARCHAR), e.TVal) AS Result, -- Use the column that is not NULL and convert to VARCHAR
        'EPIC' AS Source
    FROM dbo.labsEpic e
    JOIN vw_matching_labs ml ON e.ComponentCommonNM = ml.ComponentCommonNM
    WHERE e.LabDate BETWEEN @minDate AND @maxDate
          AND (e.NVal IS NOT NULL OR e.TVal IS NOT NULL)
    UNION ALL
    SELECT
        l.empi,
        l.LabDate,
        l.GroupCD AS Code,
        COALESCE(CAST(l.NVal AS VARCHAR), l.TVal) AS Result, -- Use the column that is not NULL and convert to VARCHAR
        'LMR' AS Source
    FROM dbo.labsLMR l
    JOIN vw_matching_labs ml ON l.GroupNM = ml.GroupNM
    WHERE l.LabDate BETWEEN @minDate AND @maxDate
          AND (l.NVal IS NOT NULL OR l.TVal IS NOT NULL)
    UNION ALL
    SELECT
        a.EMPI,
        a.LabDate,
        a.GroupCD AS Code,
        COALESCE(CAST(a.nval AS VARCHAR), a.tval) AS Result, -- Use the column that is not NULL and convert to VARCHAR
        'Archive' AS Source
    FROM dbo.labsLMRArchive a
    JOIN vw_matching_labs ml ON a.GroupNM = ml.GroupNM
    WHERE a.LabDate BETWEEN @minDate AND @maxDate
          AND (a.nval IS NOT NULL OR a.tval IS NOT NULL)
)
SELECT
    cl.EMPI,
    cl.Date,
    cl.Code,
    cl.Result,
    cl.Source
INTO dbo.Labs_old_combined
FROM CombinedLabs cl;



-- Display the top 100 records from the combined Labs table for verification
SELECT * 
FROM dbo.Labs;

-- Inspect the result set
SELECT 
    COUNT(DISTINCT EMPI) AS UniqueEMPIs,
    COUNT(DISTINCT Code) AS UniqueLabCodes
FROM dbo.Labs;







--------------------------------------------------------------------------------------------------------
--EMPIs to StudyID Crosswalk
--------------------------------------------------------------------------------------------------------


--Use NewID() for EMPI
IF OBJECT_ID('dbo.StudyID_old', 'U') IS NOT NULL DROP TABLE dbo.StudyID_old;

CREATE TABLE dbo.StudyID_old (
    EMPI VARCHAR(20) PRIMARY KEY,
    StudyID UNIQUEIDENTIFIER DEFAULT NEWID()
);

INSERT INTO dbo.StudyID_old (EMPI)
SELECT DISTINCT EMPI
FROM dbo.Labs_old;

SELECT TOP 100 *
FROM dbo.StudyID_old


--replace EMPIs with StudyID's for all three


IF OBJECT_ID('dbo.Labs_With_StudyID', 'U') IS NOT NULL DROP TABLE dbo.Labs_With_StudyID;

SELECT
    s.StudyID,
    l.Date,
    l.Code,
    l.Result,
    l.Source
INTO dbo.Labs_With_StudyID
FROM dbo.Labs l
JOIN dbo.StudyID s ON l.EMPI = s.EMPI;

-- Verify the new Labs table
SELECT TOP 100 *
FROM dbo.Labs_With_StudyID;

--renamed to Labs^



IF OBJECT_ID('dbo.Diagnoses_With_StudyID', 'U') IS NOT NULL DROP TABLE dbo.Diagnoses_With_StudyID;

SELECT
    s.StudyID,
    d.Date,
    d.Code,
    d.Code_Type,
    d.IndexDate,
    d.CodeWithType
INTO dbo.Diagnoses_With_StudyID
FROM dbo.Diagnoses d
JOIN dbo.StudyID s ON d.EMPI = s.EMPI;

-- renamed to Diagnoses


IF OBJECT_ID('dbo.Procedures_With_StudyID', 'U') IS NOT NULL DROP TABLE dbo.Procedures_With_StudyID;

SELECT
    s.StudyID,
    p.Date,
    p.Code,
    p.Code_Type,
    p.IndexDate,
    p.CodeWithType
INTO dbo.Procedures_With_StudyID
FROM dbo.Procedures p
JOIN dbo.StudyID s ON p.EMPI = s.EMPI;


-- Step 1: Create the new table with StudyID
CREATE TABLE [DM2023].[dbo].[DiabetesOutcomes_With_StudyID] (
    StudyID UNIQUEIDENTIFIER,
    [IndexDate] DATETIME2,
    [InitialA1c] FLOAT,
    [A1cGreaterThan7] INT,
    [Female] INT,
    [Married] INT,
    [GovIns] INT,
    [English] INT,
    [DaysFromIndexToInitialA1cDate] INT,
    [DaysFromIndexToA1cDateAfter12Months] INT,
    [DaysFromIndexToFirstEncounterDate] INT,
    [DaysFromIndexToLastEncounterDate] INT,
    [DaysFromIndexToLatestDate] INT,
    [DaysFromIndexToPatientTurns18] INT,
    [AgeYears] INT,
    [SDI_score] FLOAT,
    [Veteran] INT
);

-- Step 2: Insert data into the new table, replacing EMPI with StudyID
INSERT INTO [DM2023].[dbo].[DiabetesOutcomes_With_StudyID] (
    StudyID,
    [IndexDate],
    [InitialA1c],
    [A1cGreaterThan7],
    [Female],
    [Married],
    [GovIns],
    [English],
    [DaysFromIndexToInitialA1cDate],
    [DaysFromIndexToA1cDateAfter12Months],
    [DaysFromIndexToFirstEncounterDate],
    [DaysFromIndexToLastEncounterDate],
    [DaysFromIndexToLatestDate],
    [DaysFromIndexToPatientTurns18],
    [AgeYears],
    [SDI_score],
    [Veteran]
)
SELECT
    s.StudyID,
    d.[IndexDate],
    d.[InitialA1c],
    d.[A1cGreaterThan7],
    d.[Female],
    d.[Married],
    d.[GovIns],
    d.[English],
    d.[DaysFromIndexToInitialA1cDate],
    d.[DaysFromIndexToA1cDateAfter12Months],
    d.[DaysFromIndexToFirstEncounterDate],
    d.[DaysFromIndexToLastEncounterDate],
    d.[DaysFromIndexToLatestDate],
    d.[DaysFromIndexToPatientTurns18],
    d.[AgeYears],
    d.[SDI_score],
    d.[Veteran]
FROM [DM2023].[dbo].[DiabetesOutcomes] d
JOIN [DM2023].[dbo].[StudyID] s ON d.[EMPI] = s.[EMPI];

-- Verify the data in the new table
SELECT TOP 100 *
FROM [DM2023].[dbo].[DiabetesOutcomes_With_StudyID];
--renamed DiabetesOutcomes

--------------------------------------------------------------------------------------------------------
--Limit Labs dataset
--------------------------------------------------------------------------------------------------------


--- >= 1% of Labs

-- Calculate the total number of unique patients
DECLARE @TotalPatients_l INT;

SELECT @TotalPatients_l = COUNT(DISTINCT EMPI)
FROM dbo.Labs;


-- Labs found in at least 1% of the patients
IF OBJECT_ID('tempdb..#Labs_Frequency', 'U') IS NOT NULL DROP TABLE #Labs_Frequency;

SELECT Code, COUNT(DISTINCT EMPI) AS PatientCount
INTO #Labs_Frequency
FROM dbo.Labs
GROUP BY Code
HAVING COUNT(DISTINCT EMPI) >= 0.01 * @TotalPatients_l;

-- Verify the temporary table
SELECT * 
FROM #Labs_Frequency;

-- Labs dataset with StudyID and filtered codes
IF OBJECT_ID('dbo.Labs_limited', 'U') IS NOT NULL DROP TABLE dbo.Labs_limited;

SELECT
    s.StudyID,
    l.Date,
    l.Code,
    l.Result,
    l.Source
INTO dbo.Labs_limited
FROM dbo.Labs l
JOIN dbo.StudyID s ON l.EMPI = s.EMPI
JOIN #Labs_Frequency lf ON l.Code = lf.Code;

-- Verify the Labs_limited table
SELECT TOP 100 *
FROM dbo.Labs_limited;


--------------------------------------------------------------------------------------------------------
--Limit Diagnoses dataset
--------------------------------------------------------------------------------------------------------


-- >= 1% of Diagnoses

DECLARE @TotalPatients_d INT;

SELECT @TotalPatients_d = COUNT(DISTINCT EMPI)
FROM dbo.Diagnoses;  

-- Diagnoses found in at least 1% of the patients
IF OBJECT_ID('tempdb..#Diagnoses_Frequency', 'U') IS NOT NULL DROP TABLE #Diagnoses_Frequency;

SELECT CodeWithType, COUNT(DISTINCT EMPI) AS PatientCount
INTO #Diagnoses_Frequency
FROM dbo.Diagnoses
GROUP BY CodeWithType
HAVING COUNT(DISTINCT EMPI) >= 0.01 * @TotalPatients_d;

-- Verify the temporary table
SELECT * 
FROM #Diagnoses_Frequency;

-- Diagnoses dataset with StudyID and filtered codes
IF OBJECT_ID('dbo.Diagnoses_limited', 'U') IS NOT NULL DROP TABLE dbo.Diagnoses_limited;

SELECT
    s.StudyID,
    d.Date,
    d.Code,
    d.Code_Type,
    d.IndexDate,
    d.CodeWithType
INTO dbo.Diagnoses_limited
FROM dbo.Diagnoses d
JOIN dbo.StudyID s ON d.EMPI = s.EMPI
JOIN #Diagnoses_Frequency df ON d.CodeWithType = df.CodeWithType;

-- Verify the Diagnoses_limited table
SELECT TOP 100 *
FROM dbo.Diagnoses_limited;

--------------------------------------------------------------------------------------------------------
--Limit Procedures dataset
--------------------------------------------------------------------------------------------------------


--- >= 1% of Procedures

DECLARE @TotalPatients_p INT;

SELECT @TotalPatients_p = COUNT(DISTINCT EMPI)
FROM dbo.Procedures;  

-- Procedures found in at least 1% of the patients
IF OBJECT_ID('tempdb..#Procedures_Frequency', 'U') IS NOT NULL DROP TABLE #Procedures_Frequency;

SELECT CodeWithType, COUNT(DISTINCT EMPI) AS PatientCount
INTO #Procedures_Frequency
FROM dbo.Procedures
GROUP BY CodeWithType
HAVING COUNT(DISTINCT EMPI) >= 0.01 * @TotalPatients_p;

-- Verify the temporary table
SELECT * 
FROM #Procedures_Frequency;


-- Procedures dataset with StudyID and filtered codes
IF OBJECT_ID('dbo.Procedures_limited', 'U') IS NOT NULL DROP TABLE dbo.Procedures_limited;

SELECT
    s.StudyID,
    p.Date,
    p.Code,
    p.Code_Type,
    p.IndexDate,
    p.CodeWithType
INTO dbo.Procedures_limited
FROM dbo.Procedures p
JOIN dbo.StudyID s ON p.EMPI = s.EMPI
JOIN #Procedures_Frequency pf ON p.CodeWithType = pf.CodeWithType;

-- Verify the Procedures_limited table
SELECT TOP 100 *
FROM dbo.Procedures_limited;

--------------------------------------------------------------------------------------------------------
-- 3 years prior to StudyEntry: Diagnoses, Procedures, Labs
--------------------------------------------------------------------------------------------------------


-- Create Diagnoses_3year table with data within 3 years prior to IndexDate
IF OBJECT_ID('dbo.Diagnoses_3year', 'U') IS NOT NULL DROP TABLE dbo.Diagnoses_3year;

SELECT
    d.StudyID,
    d.Date,
    d.Code,
    d.Code_Type,
    d.IndexDate,
    d.CodeWithType
INTO dbo.Diagnoses_3year
FROM dbo.Diagnoses_limited d
WHERE d.Date BETWEEN DATEADD(YEAR, -3, d.IndexDate) AND d.IndexDate;

-- Verify the Diagnoses_3year table
SELECT TOP 100 *
FROM dbo.Diagnoses_3year;




-- Create Procedures_3year table with data within 3 years prior to IndexDate
IF OBJECT_ID('dbo.Procedures_3year', 'U') IS NOT NULL DROP TABLE dbo.Procedures_3year;

SELECT
    p.StudyID,
    p.Date,
    p.Code,
    p.Code_Type,
    p.IndexDate,
    p.CodeWithType
INTO dbo.Procedures_3year
FROM dbo.Procedures_limited p
WHERE p.Date BETWEEN DATEADD(YEAR, -3, p.IndexDate) AND p.IndexDate;

-- Verify the Procedures_3year table
SELECT TOP 100 *
FROM dbo.Procedures_3year;



-- Consolidate IndexDate from both Procedures_limited and Diagnoses_limited
IF OBJECT_ID('tempdb..#Consolidated_IndexDate', 'U') IS NOT NULL DROP TABLE #Consolidated_IndexDate;

SELECT DISTINCT
    StudyID,
    IndexDate
INTO #Consolidated_IndexDate
FROM (
    SELECT StudyID, IndexDate FROM dbo.Procedures_limited
    UNION
    SELECT StudyID, IndexDate FROM dbo.Diagnoses_limited
) AS Consolidated;

-- Verify the Consolidated IndexDate table
SELECT TOP 100 *
FROM #Consolidated_IndexDate;



-- Create Labs_3year table with data within 3 years prior to any IndexDate from Procedures_limited or Diagnoses_limited
IF OBJECT_ID('dbo.Labs_3year', 'U') IS NOT NULL DROP TABLE dbo.Labs_3year;

SELECT
    l.StudyID,
    l.Date,
    l.Code,
    l.Result,
    l.Source,
    c.IndexDate
INTO dbo.Labs_3year
FROM dbo.Labs_limited l
JOIN #Consolidated_IndexDate c ON l.StudyID = c.StudyID
WHERE l.Date BETWEEN DATEADD(YEAR, -3, c.IndexDate) AND c.IndexDate;

-- Verify the Labs_3year table
SELECT TOP 100 *
FROM dbo.Labs_3year;





-------------------------------------------------------------------------------------------------------
-- Replacing ICD9 with ICD10 (Procedures)
--------------------------------------------------------------------------------------------------------
-- includes lots of debugging/investigating
-- crosswalk creation, #CTE_Procedures, Procedures_ICD10



--EXEC sp_rename 'Procedures', 'Procedures_ICD9_ICD10';

-- Creating a new Procedures table with ICD9 codes replaced by ICD10 codes and updating CodeWithType column
SELECT 
    p.StudyID,
    p.Date,
    COALESCE(c.ICD10, p.Code) AS Code,
    CASE 
        WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
        ELSE p.Code_Type
    END AS Code_Type,
    p.IndexDate,
    CONCAT(
        COALESCE(c.ICD10, p.Code), '_', 
        CASE 
            WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
            ELSE p.Code_Type
        END
    ) AS CodeWithType
INTO Procedures
FROM Procedures_ICD9_ICD10 p
LEFT JOIN DM2023..ICD9toICD10ProcCrosswalkSimple c
ON p.Code = c.ICD9
WHERE p.Code_Type = 'ICD9' OR p.Code LIKE '%.%';



-- Validate the new Procedures table
SELECT * FROM Procedures;


-- Create a view for the updated Procedures table
CREATE VIEW Updated_Procedures AS
SELECT 
    p.StudyID,
    p.Date,
    COALESCE(c.ICD10, p.Code) AS Code,
    CASE 
        WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
        ELSE p.Code_Type
    END AS Code_Type,
    p.IndexDate,
    CONCAT(
        COALESCE(c.ICD10, p.Code), '_', 
        CASE 
            WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
            ELSE p.Code_Type
        END
    ) AS CodeWithType
FROM Procedures_ICD9_ICD10 p
LEFT JOIN ICD9toICD10ProcCrosswalkSimple c
ON p.Code = c.ICD9
WHERE p.Code_Type = 'ICD9'
UNION ALL
SELECT 
    p.StudyID,
    p.Date,
    p.Code,
    p.Code_Type,
    p.IndexDate,
    p.CodeWithType
FROM Procedures_ICD9_ICD10 p
WHERE p.Code_Type != 'ICD9';



-- Query the view to verify the updated procedures data
SELECT * FROM Updated_Procedures;


CREATE VIEW Procedures_ICD9_ICD10_Updated AS
SELECT 
    p.StudyID,
    p.Date,
    COALESCE(c.ICD10, p.Code) AS Code,
    CASE 
        WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
        ELSE p.Code_Type
    END AS Code_Type,
    p.IndexDate,
    CASE 
        WHEN c.ICD10 IS NOT NULL THEN CONCAT(c.ICD10, '_ICD10')
        ELSE p.CodeWithType
    END AS CodeWithType
FROM 
    Procedures_ICD9_ICD10 p
LEFT JOIN 
    ICD9toICD10ProcCrosswalkSimple c
ON 
    p.Code = c.ICD9
WHERE 
    p.Code_Type = 'ICD9'

UNION ALL

SELECT 
    p.StudyID,
    p.Date,
    p.Code,
    p.Code_Type,
    p.IndexDate,
    p.CodeWithType
FROM 
    Procedures_ICD9_ICD10 p
WHERE 
    p.Code_Type <> 'ICD9';



SELECT * FROM Procedures_ICD9_ICD10_Updated;



--debugging

SELECT 
    DISTINCT p.Code
FROM 
    Procedures_ICD9_ICD10 p
LEFT JOIN 
    ICD9toICD10ProcCrosswalkSimple c
ON 
    p.Code = c.ICD9
WHERE 
    p.Code_Type = 'ICD9' 
    AND c.ICD9 IS NULL;




SELECT 
    p.Code AS ICD9_Code,
    CASE 
        WHEN c.ICD9 IS NOT NULL THEN 'Present in Crosswalk'
        ELSE 'Missing in Crosswalk'
    END AS Crosswalk_Status
FROM 
    Procedures_ICD9_ICD10 p
LEFT JOIN 
    ICD9toICD10ProcCrosswalkSimple c
ON 
    p.Code = c.ICD9
WHERE 
    p.Code_Type = 'ICD9';




-- 1. Count of ICD9 Codes in Procedures_ICD9_ICD10
-- 109264
SELECT COUNT(*) AS Total_ICD9_Codes
FROM Procedures_ICD9_ICD10
WHERE Code_Type = 'ICD9';



-- 3. Count of ICD9 Codes in Procedures_ICD9_ICD10 with Matching Crosswalk Entries
-- 9548
SELECT COUNT(*) AS Matching_ICD9_Codes
FROM Procedures_ICD9_ICD10 p
INNER JOIN ICD9toICD10ProcCrosswalkSimple c
ON p.Code = c.ICD9
WHERE p.Code_Type = 'ICD9';

-- 4. Count of ICD9 Codes in Procedures_ICD9_ICD10 without Matching Crosswalk Entries
-- 99716
SELECT COUNT(*) AS Missing_ICD9_Codes
FROM Procedures_ICD9_ICD10 p
LEFT JOIN ICD9toICD10ProcCrosswalkSimple c
ON p.Code = c.ICD9
WHERE p.Code_Type = 'ICD9' 
AND c.ICD9 IS NULL;



-- Number of Unique ICD9 Codes in Procedures_ICD9_ICD10
-- 762
SELECT COUNT(DISTINCT Code) AS Unique_ICD9_Codes
FROM Procedures_ICD9_ICD10
WHERE Code_Type = 'ICD9';

-- Number of Unique ICD9 Codes in ICD9toICD10ProcCrosswalkSimple
-- 3878
SELECT COUNT(DISTINCT ICD9) AS Unique_ICD9_Codes_Crosswalk
FROM ICD9toICD10ProcCrosswalkSimple;

--  Number of Unique ICD9 Codes in Crosswalk Table that Do Not Match with Procedures_ICD9_ICD10
-- 3717
-- 3878 - 161 matching = 3717
SELECT COUNT(DISTINCT c.ICD9) AS Non_Matching_ICD9_Codes_Crosswalk
FROM ICD9toICD10ProcCrosswalkSimple c
LEFT JOIN Procedures_ICD9_ICD10 p
ON REPLACE(TRIM(p.Code), '.', '') = REPLACE(TRIM(c.ICD9), '.', '')
WHERE p.Code IS NULL;

-- Number of Unique ICD9 Codes in Procedures_ICD9_ICD10 with Matching Crosswalk Entries
-- 161
SELECT COUNT(DISTINCT p.Code) AS Matching_ICD9_Codes
FROM Procedures_ICD9_ICD10 p
INNER JOIN ICD9toICD10ProcCrosswalkSimple c
ON p.Code = c.ICD9
WHERE p.Code_Type = 'ICD9';

-- Number of Unique ICD9 Codes in Procedures_ICD9_ICD10 without Matching Crosswalk Entries
-- 601
SELECT COUNT(DISTINCT p.Code) AS Missing_ICD9_Codes
FROM Procedures_ICD9_ICD10 p
LEFT JOIN ICD9toICD10ProcCrosswalkSimple c
ON p.Code = c.ICD9
WHERE p.Code_Type = 'ICD9' 
AND c.ICD9 IS NULL;


-- Normalize codes in Procedures_ICD9_ICD10 and compare with normalized crosswalk codes
SELECT DISTINCT p.Code AS Original_ICD9_Code, 
                REPLACE(p.Code, '.', '') AS Normalized_ICD9_Code,
                c.ICD9 AS Crosswalk_ICD9_Code,
                REPLACE(c.ICD9, '.', '') AS Normalized_Crosswalk_ICD9_Code
FROM Procedures_ICD9_ICD10 p
LEFT JOIN ICD9toICD10ProcCrosswalkSimple c
ON REPLACE(p.Code, '.', '') = REPLACE(c.ICD9, '.', '')
WHERE p.Code_Type = 'ICD9'
AND c.ICD9 IS NULL;


-- Check the lengths of ICD9 codes in Procedures_ICD9_ICD10
SELECT DISTINCT LENGTH(Code) AS Code_Length, COUNT(*) AS Code_Count
FROM Procedures_ICD9_ICD10
WHERE Code_Type = 'ICD9'
GROUP BY LENGTH(Code);

-- Check the lengths of ICD9 codes in ICD9toICD10ProcCrosswalkSimple
SELECT DISTINCT LENGTH(ICD9) AS Code_Length, COUNT(*) AS Code_Count
FROM ICD9toICD10ProcCrosswalkSimple
GROUP BY LENGTH(ICD9);



SELECT TOP(10)
    p.Code AS Original_ICD9_Code,
    TRIM(p.Code) AS Trimmed_ICD9_Code,
    REPLACE(TRIM(p.Code), '.', '') AS Normalized_ICD9_Code,
    c.ICD9 AS Crosswalk_ICD9_Code,
    TRIM(c.ICD9) AS Trimmed_Crosswalk_ICD9_Code,
    REPLACE(TRIM(c.ICD9), '.', '') AS Normalized_Crosswalk_ICD9_Code,
    CASE 
        WHEN p.Code <> TRIM(p.Code) THEN 'Leading/Trailing Spaces in Procedures Table'
        WHEN c.ICD9 <> TRIM(c.ICD9) THEN 'Leading/Trailing Spaces in Crosswalk Table'
        WHEN REPLACE(p.Code, '.', '') <> REPLACE(c.ICD9, '.', '') THEN 'Formatting Difference (Decimal Points or Other)'
        ELSE 'Other Mismatch'
    END AS Mismatch_Reason
FROM 
    Procedures_ICD9_ICD10 p
LEFT JOIN 
    ICD9toICD10ProcCrosswalkSimple c
ON 
    REPLACE(TRIM(p.Code), '.', '') = REPLACE(TRIM(c.ICD9), '.', '')
WHERE 
    p.Code_Type = 'ICD9'
    AND c.ICD9 IS NULL


-- Query to print all codes that match up perfectly
-- 9548 total matching 
-- 161 distinct
SELECT 
    p.Code AS Original_ICD9_Code,
    TRIM(p.Code) AS Trimmed_ICD9_Code,
    REPLACE(TRIM(p.Code), '.', '') AS Normalized_ICD9_Code,
    c.ICD9 AS Crosswalk_ICD9_Code,
    TRIM(c.ICD9) AS Trimmed_Crosswalk_ICD9_Code,
    REPLACE(TRIM(c.ICD9), '.', '') AS Normalized_Crosswalk_ICD9_Code,
    'Perfect Match' AS Match_Status
FROM 
    Procedures_ICD9_ICD10 p
INNER JOIN 
    ICD9toICD10ProcCrosswalkSimple c
ON 
    REPLACE(TRIM(p.Code), '.', '') = REPLACE(TRIM(c.ICD9), '.', '')
WHERE 
    p.Code_Type = 'ICD9'


-- Query to print all unique codes that do not match up
-- 601
SELECT 
    DISTINCT p.Code AS Original_ICD9_Code,
    TRIM(p.Code) AS Trimmed_ICD9_Code,
    REPLACE(TRIM(p.Code), '.', '') AS Normalized_ICD9_Code
FROM 
    Procedures_ICD9_ICD10 p
LEFT JOIN 
    ICD9toICD10ProcCrosswalkSimple c
ON 
    REPLACE(TRIM(p.Code), '.', '') = REPLACE(TRIM(c.ICD9), '.', '')
WHERE 
    p.Code_Type = 'ICD9'
    AND c.ICD9 IS NULL;


-- Number of Unique ICD9 codes in crosswalk
-- 3878
SELECT COUNT(DISTINCT ICD9) AS Unique_ICD9_Codes_Crosswalk
FROM ICD9toICD10ProcCrosswalkSimple;




-- Combine diagnosis and procedure crosswalk tables
IF OBJECT_ID('tempdb..#tmp_ComboCrosswalk', 'U') IS NOT NULL DROP TABLE #tmp_ComboCrosswalk;

SELECT *
INTO #tmp_ComboCrosswalk
FROM DM2023..ICD9ToICD10DxCrosswalkSimple
UNION
SELECT *
FROM DM2023..ICD9ToICD10ProcCrosswalkSimple;



---- Create the final Procedures table with updated codes
--IF OBJECT_ID('dbo.Procedures', 'U') IS NOT NULL DROP TABLE dbo.Procedures;

--SELECT
--    prc.StudyID,
--    prc.Date,
--    CASE
--        WHEN prc.Code_Type = 'ICD9' AND cw.ICD10 IS NOT NULL THEN cw.ICD10
--        ELSE prc.Code
--    END AS Code,
--    CASE
--        WHEN prc.Code_Type = 'ICD9' AND cw.ICD10 IS NOT NULL THEN 'ICD10'
--        ELSE prc.Code_Type
--    END AS Code_Type,
--    prc.IndexDate,
--    CASE
--        WHEN prc.Code_Type = 'ICD9' AND cw.ICD10 IS NOT NULL THEN cw.ICD10 + '_ICD10'
--        ELSE prc.CodeWithType
--    END AS CodeWithType
--INTO dbo.Procedures
--FROM #CTE_Procedures prc
--LEFT JOIN #tmp_ComboCrosswalk cw
--ON REPLACE(TRIM(prc.Code), '.', '') = REPLACE(TRIM(cw.ICD9), '.', '');


--SELECT TOP 1000 *
--FROM dbo.Procedures;


---- Correct replacement process and trim resulting ICD10 codes
--IF OBJECT_ID('dbo.Procedures', 'U') IS NOT NULL DROP TABLE dbo.Procedures;

--SELECT
--    prc.StudyID,
--    prc.Date,
--    CASE
--        WHEN prc.Code_Type = 'ICD9' AND cw.ICD10 IS NOT NULL THEN 
--            CASE 
--                WHEN CHARINDEX('.', cw.ICD10) > 0 THEN LEFT(cw.ICD10, CHARINDEX('.', cw.ICD10)) + SUBSTRING(cw.ICD10, CHARINDEX('.', cw.ICD10) + 1, 1)
--                ELSE cw.ICD10
--            END
--        ELSE prc.Code
--    END AS Code,
--    CASE
--        WHEN prc.Code_Type = 'ICD9' AND cw.ICD10 IS NOT NULL THEN 'ICD10'
--        ELSE prc.Code_Type
--    END AS Code_Type,
--    prc.IndexDate,
--    CASE
--        WHEN prc.Code_Type = 'ICD9' AND cw.ICD10 IS NOT NULL THEN 
--            CASE 
--                WHEN CHARINDEX('.', cw.ICD10) > 0 THEN LEFT(cw.ICD10, CHARINDEX('.', cw.ICD10)) + SUBSTRING(cw.ICD10, CHARINDEX('.', cw.ICD10) + 1, 1)
--                ELSE cw.ICD10
--            END + '_ICD10'
--        ELSE prc.CodeWithType
--    END AS CodeWithType
--INTO dbo.Procedures
--FROM #CTE_Procedures prc
--LEFT JOIN ICD9toICD10ProcCrosswalkSimple cw
--ON REPLACE(TRIM(prc.Code), '.', '') = REPLACE(TRIM(cw.ICD9), '.', '');






-- Count of ICD9 Codes Remaining
SELECT COUNT(DISTINCT Code) AS ICD9Count
FROM dbo.Procedures
WHERE Code_Type = 'ICD9';

-- List of ICD9 Codes Remaining
SELECT DISTINCT(Code) As ICD9left
FROM #Procedures_ICD10
WHERE Code_Type = 'ICD9';

-- Identify ICD9 Codes That Were Not Replaced
SELECT DISTINCT prc.Code AS ICD9_Code
FROM dbo.Procedures prc
LEFT JOIN #tmp_ComboCrosswalk cw
ON REPLACE(TRIM(prc.Code), '.', '') = REPLACE(TRIM(cw.ICD9), '.', '')
WHERE prc.Code_Type = 'ICD9'
AND cw.ICD10 IS NULL;

-- Verify ICD9 to ICD10 Mapping
SELECT DISTINCT prc.Code AS ICD9_Code, cw.ICD10 AS ICD10_Code
FROM dbo.Procedures prc
LEFT JOIN #tmp_ComboCrosswalk cw
ON REPLACE(TRIM(prc.Code), '.', '') = REPLACE(TRIM(cw.ICD9), '.', '')
WHERE prc.Code_Type = 'ICD9';

-- Count of Replaced ICD9 Codes with ICD10 Codes
SELECT COUNT(*) AS ReplacedICD9Count
FROM dbo.Procedures prc
INNER JOIN #tmp_ComboCrosswalk cw
ON REPLACE(TRIM(prc.Code), '.', '') = REPLACE(TRIM(cw.ICD9), '.', '')
WHERE prc.Code_Type = 'ICD10';

-- List of Successfully Replaced ICD9 Codes
SELECT DISTINCT prc.Code AS ICD9_Code, cw.ICD10 AS ICD10_Code
FROM dbo.Procedures prc
INNER JOIN #tmp_ComboCrosswalk cw
ON REPLACE(TRIM(prc.Code), '.', '') = REPLACE(TRIM(cw.ICD9), '.', '')
WHERE prc.Code_Type = 'ICD10';


-- Check if the missing ICD9 codes are present in the crosswalk table
SELECT DISTINCT p.Code AS Missing_ICD9_Codes
FROM Procedures_ICD9_ICD10 p
LEFT JOIN ICD9toICD10ProcCrosswalkSimple c
ON REPLACE(TRIM(p.Code), '.', '') = REPLACE(TRIM(c.ICD9), '.', '')
WHERE p.Code_Type = 'ICD9'
AND c.ICD10 IS NULL
AND p.Code IN ('02.2', '32.3', '32.4', '33.5', '36.01', '36.02', '36.05', 
'37.4', '39.8', '45.8', '47.1', '53.7', '54.5', '56.3', '58.9', '60.2', 
'65.4', '65.8', '67.5', '68.5', '68.6', '77.96', '81.09', '85.7');


-- Verify the original ICD9 codes before any trimming
SELECT DISTINCT p.Code AS Original_ICD9_Codes, c.ICD10
FROM Procedures_ICD9_ICD10 p
LEFT JOIN ICD9toICD10ProcCrosswalkSimple c
ON REPLACE(TRIM(p.Code), '.', '') = REPLACE(TRIM(c.ICD9), '.', '')
WHERE p.Code_Type = 'ICD9';


SELECT TOP 100 * FROM dbo.Procedures_ICD10;


-- Step 1: Create a new table for manual mappings
IF OBJECT_ID('dbo.ManualICD9ToICD10ProcCrosswalk', 'U') IS NOT NULL DROP TABLE dbo.ManualICD9ToICD10ProcCrosswalk;

CREATE TABLE dbo.ManualICD9ToICD10ProcCrosswalk (
    ICD9 VARCHAR(10),
    ICD10 VARCHAR(10)
);

-- Step 2: Insert manual mappings into the new table
INSERT INTO dbo.ManualICD9ToICD10ProcCrosswalk (ICD9, ICD10)
VALUES
('36.02', '027034Z'),
('53.7', '0BQR0ZZ'),
('37.4', '02Q50ZZ'),
('54.5', '0DN80ZZ'),
('45.8', '0DTE0ZZ'),
('33.5', '0BYC0Z0'),
('67.5', '0UQG0ZZ'),
('68.5', '0UT90ZZ'),
('65.8', '0U580ZZ'),
('60.2', '0VT07ZZ'),
('39.8', '0G973ZX'),
('36.05', '02713DZ'),
('81.09', '0SG10AJ'),
('2.2', '00940ZZ'),
('85.7', '0H0T0ZZ'),
('65.4', '0UT90ZZ'),
('36.01', '02703DZ'),
('32.3', '0BBC0ZZ'),
('32.4', '0BT40ZZ'),
('47.1', '0DTJ0ZZ'),
('58.9', '0TSS0ZZ'),
('68.6', '0UT90ZZ'),
('65.00', '0UT90ZZ'),
('47.00', '0DTJ0ZZ'),
('68.30', '0UT80ZZ'),
('68.40', '0UT94ZZ'),
('56.3', '0TJB0ZZ');

-- Step 3: Combine the existing crosswalk with the manual mappings
IF OBJECT_ID('dbo.ICD9ToICD10ProcCrosswalkUpdated', 'U') IS NOT NULL DROP TABLE dbo.ICD9ToICD10ProcCrosswalkUpdated;

SELECT *
INTO dbo.ICD9ToICD10ProcCrosswalkUpdated
FROM (
    SELECT ICD9, ICD10
    FROM ICD9ToICD10ProcCrosswalkSimple
    UNION
    SELECT ICD9, ICD10
    FROM dbo.ManualICD9ToICD10ProcCrosswalk
) AS CombinedCrosswalk;

-- Step 4: Verify the new crosswalk table
SELECT * FROM dbo.ICD9ToICD10ProcCrosswalkUpdated;


-- Step 5: Identify ICD9 codes in Procedures_ICD9_ICD10 that are not in the crosswalk
SELECT DISTINCT p.Code AS ICD9_Code
FROM dbo.Procedures_ICD9_ICD10 p
WHERE p.Code_Type = 'ICD9'
AND NOT EXISTS (
    SELECT 1
    FROM dbo.ICD9ToICD10ProcCrosswalkUpdated c
    WHERE p.Code = c.ICD9
);



-- Create a temporary table to store the initial processing
IF OBJECT_ID('tempdb..#CTE_Procedures', 'U') IS NOT NULL DROP TABLE #CTE_Procedures;

;WITH CTE_Procedures AS (
    SELECT
        s.StudyID,
        prc.Date,
        CASE 
            WHEN prc.Code_Type = 'ICD9' THEN 
                RIGHT('0000' + LEFT(prc.Code, CHARINDEX('.', prc.Code) - 1), 2) + '.' +
                LEFT(RIGHT(prc.Code, LEN(prc.Code) - CHARINDEX('.', prc.Code)), 2) + 
                REPLICATE('0', 2 - LEN(RIGHT(prc.Code, LEN(prc.Code) - CHARINDEX('.', prc.Code))))
            ELSE prc.Code 
        END AS Code,
        prc.Code_Type,
        po.IndexDate,
        CASE WHEN prc.Date <= po.IndexDate THEN 1 ELSE 0 END AS ProcedureBeforeOrOnIndexDate,
        CASE 
            WHEN prc.Code_Type = 'ICD9' THEN 
                RIGHT('0000' + LEFT(prc.Code, CHARINDEX('.', prc.Code) - 1), 2) + '.' +
                LEFT(RIGHT(prc.Code, LEN(prc.Code) - CHARINDEX('.', prc.Code)), 2) + 
                REPLICATE('0', 2 - LEN(RIGHT(prc.Code, LEN(prc.Code) - CHARINDEX('.', prc.Code)))) + 
                '_' + prc.Code_Type
            ELSE prc.Code + '_' + prc.Code_Type
        END AS CodeWithType
    FROM prc_2_pcp_combined prc
    INNER JOIN StudyID s ON prc.EMPI = s.EMPI
    INNER JOIN DiabetesOutcomes po ON s.StudyID = po.StudyID
    WHERE prc.Code_Type IN ('CPT', 'ICD9', 'ICD10') AND prc.Code IS NOT NULL AND prc.Code <> ''
),
AggregatedProcedures AS (
    SELECT
        StudyID,
        Code,
        Code_Type,
        MAX(Date) AS LatestDate,
        MAX(IndexDate) AS LatestIndexDate,
        CodeWithType
    FROM CTE_Procedures
    WHERE ProcedureBeforeOrOnIndexDate = 1
    GROUP BY StudyID, Code, Code_Type, CodeWithType
)

SELECT 
    StudyID,
    LatestDate AS Date,
    Code,
    Code_Type,
    LatestIndexDate AS IndexDate,
    CodeWithType
INTO #CTE_Procedures
FROM AggregatedProcedures;

SELECT COUNT(DISTINCT Code) FROM #CTE_Procedures
WHERE Code_Type = 'ICD9';

SELECT TOP 1000 * FROM #CTE_Procedures;







-- Identify ICD9 codes in Procedures_ICD9_ICD10 that are not in the crosswalk
SELECT DISTINCT p.Code
FROM #CTE_Procedures p
LEFT JOIN dbo.ICD9ToICD10ProcCrosswalkUpdated c
    ON p.Code_Type = 'ICD9' 
    AND p.Code = c.ICD9
WHERE p.Code_Type = 'ICD9' AND c.ICD10 IS NULL;



-- Step 1: Create a temporary table to store the formatted crosswalk
IF OBJECT_ID('tempdb..#FormattedICD9ToICD10Crosswalk', 'U') IS NOT NULL DROP TABLE #FormattedICD9ToICD10Crosswalk;

-- Step 2: Insert the formatted crosswalk entries into the temporary table
SELECT
    RIGHT('0000' + LEFT(ICD9, CHARINDEX('.', ICD9) - 1), 2) + '.' +
    LEFT(RIGHT(ICD9, LEN(ICD9) - CHARINDEX('.', ICD9)), 2) + 
    REPLICATE('0', 2 - LEN(RIGHT(ICD9, LEN(ICD9) - CHARINDEX('.', ICD9)))) AS ICD9,
    ICD10
INTO #FormattedICD9ToICD10Crosswalk
FROM dbo.ICD9ToICD10ProcCrosswalkUpdated;

-- Step 3: Verify the formatted crosswalk
SELECT * FROM #FormattedICD9ToICD10Crosswalk;

-- Step 4: Replace the existing crosswalk with the formatted one
IF OBJECT_ID('dbo.ICD9ToICD10ProcCrosswalkFormatted', 'U') IS NOT NULL DROP TABLE dbo.ICD9ToICD10ProcCrosswalkFormatted;

SELECT *
INTO dbo.ICD9ToICD10ProcCrosswalkFormatted
FROM #FormattedICD9ToICD10Crosswalk;

-- Step 5: Verify the new crosswalk table
SELECT * FROM dbo.ICD9ToICD10ProcCrosswalkFormatted;



-- Step 3: Replace ICD9 codes in Procedures_ICD9_ICD10 with corresponding ICD10 codes from the crosswalk
IF OBJECT_ID('Procedures_ICD10', 'U') IS NOT NULL DROP TABLE Procedures_ICD10;

SELECT
    p.StudyID,
    p.Date,
    ISNULL(c.ICD10, p.Code) AS Code,
    CASE 
        WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
        ELSE p.Code_Type
    END AS Code_Type,
    p.IndexDate,
    ISNULL(c.ICD10, p.Code) + '_' + 
    CASE 
        WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
        ELSE p.Code_Type
    END AS CodeWithType
INTO Procedures_ICD10
FROM #CTE_Procedures p
LEFT JOIN dbo.ICD9ToICD10ProcCrosswalkFormatted c
    ON p.Code_Type = 'ICD9' 
    AND p.Code = c.ICD9;

-- Step 4: Verify the new Procedures_ICD10 table
SELECT TOP 1000 * FROM #Procedures_ICD10;

-- List of ICD9 Codes Remaining
SELECT DISTINCT(Code) As ICD9left
FROM #Procedures_ICD10
WHERE Code_Type = 'ICD9';
--0 left, all ICD10




-------------------------------------------------------------------------------------------------------
-- Replacing ICD9 with ICD10 (Diagnoses)
--------------------------------------------------------------------------------------------------------



-- Step 1: Create a temporary table with StudyID and full Codes
IF OBJECT_ID('tempdb..#CTE_Diagnoses', 'U') IS NOT NULL DROP TABLE #CTE_Diagnoses;

;WITH CTE_Diagnoses AS (
    SELECT
        s.StudyID,
        dia.Date,
        dia.Code,
        dia.Code_Type,
        do.IndexDate,
        CASE WHEN dia.Date <= do.IndexDate THEN 1 ELSE 0 END AS DiagnosisBeforeOrOnIndexDate,
        dia.Code + '_' + dia.Code_Type AS CodeWithType
    FROM dia_2_pcp_combined dia
    INNER JOIN StudyID s ON dia.EMPI = s.EMPI
    INNER JOIN DiabetesOutcomes do ON s.StudyID = do.StudyID
    WHERE dia.Code_Type IN ('ICD9', 'ICD10') AND dia.Code IS NOT NULL AND dia.Code <> ''
),
AggregatedDiagnoses AS (
    SELECT
        StudyID,
        Code,
        Code_Type,
        MAX(Date) AS LatestDate,
        MAX(IndexDate) AS LatestIndexDate,
        CodeWithType
    FROM CTE_Diagnoses
    WHERE DiagnosisBeforeOrOnIndexDate = 1
    GROUP BY StudyID, Code, Code_Type, CodeWithType
)

SELECT 
    StudyID,
    LatestDate AS Date,
    Code,
    Code_Type,
    LatestIndexDate AS IndexDate,
    CodeWithType
INTO #CTE_Diagnoses
FROM AggregatedDiagnoses;

SELECT *
FROM #CTE_Diagnoses


-- Step 2: Create a new Diagnoses_ICD10 table with ICD9 codes replaced by ICD10 codes
IF OBJECT_ID('dbo.Diagnoses_ICD10', 'U') IS NOT NULL DROP TABLE dbo.Diagnoses_ICD10;

-- Replace ICD9 codes with ICD10 codes
SELECT 
    d.StudyID,
    d.Date,
    COALESCE(c.ICD10, d.Code) AS Code,
    CASE 
        WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
        ELSE d.Code_Type
    END AS Code_Type,
    d.IndexDate,
    COALESCE(c.ICD10, d.Code) + '_' + CASE 
                                         WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
                                         ELSE d.Code_Type
                                       END AS CodeWithType
INTO dbo.Diagnoses_ICD10
FROM #CTE_Diagnoses d
LEFT JOIN ICD9ToICD10DxCrosswalkSimple c
ON d.Code = c.ICD9
WHERE d.Code_Type = 'ICD9' OR d.Code_Type = 'ICD10';





-- Step 3: Print the number of ICD9 codes initially
-- 2179839
SELECT COUNT(*) AS Initial_ICD9_Count
FROM #CTE_Diagnoses
WHERE Code_Type = 'ICD9';

--11678
SELECT COUNT(Distinct Code) AS Initial_Unique_ICD9_Count
FROM #CTE_Diagnoses
WHERE Code_Type = 'ICD9';

-- Step 4: Print the number of ICD9 codes remaining after using the crosswalk
-- 377650
SELECT COUNT(*) AS Remaining_ICD9_Count
FROM dbo.Diagnoses_ICD10
WHERE Code_Type = 'ICD9';

-- Step 5: List the ICD9 codes remaining after using the crosswalk
SELECT DISTINCT Code AS Remaining_ICD9_Codes
FROM dbo.Diagnoses_ICD10
WHERE Code_Type = 'ICD9';

-- Step 6: Print the number of distinct ICD9 codes remaining
-- 2277
SELECT COUNT(DISTINCT Code) AS Distinct_Remaining_ICD9_Codes
FROM dbo.Diagnoses_ICD10
WHERE Code_Type = 'ICD9';


-- investigation

-- Step 1: Create a temporary table with the remaining ICD9 codes from Diagnoses_ICD10
IF OBJECT_ID('tempdb..#Remaining_ICD9_Codes', 'U') IS NOT NULL DROP TABLE #Remaining_ICD9_Codes;

SELECT DISTINCT Code AS ICD9code
INTO #Remaining_ICD9_Codes
FROM dbo.Diagnoses_ICD10
WHERE Code_Type = 'ICD9';

-- Step 2: Join the temporary table with the ICD9 table to get the ICD9 descriptions
IF OBJECT_ID('tempdb..#ICD9_With_Description', 'U') IS NOT NULL DROP TABLE #ICD9_With_Description;

SELECT r.ICD9code, i9.Description AS ICD9Description
INTO #ICD9_With_Description
FROM #Remaining_ICD9_Codes r
INNER JOIN ICD9 i9 ON r.ICD9code = i9.ICD9code;

-- Step 3: Join the result with the ICD10 table to get the matching ICD10 descriptions
IF OBJECT_ID('tempdb..#ICD9_ICD10_Match', 'U') IS NOT NULL DROP TABLE #ICD9_ICD10_Match;

SELECT 
    i9.ICD9code,
    i9.ICD9Description,
    i10.ICD10code,
    i10.Description AS ICD10Description
INTO #ICD9_ICD10_Match
FROM #ICD9_With_Description i9
INNER JOIN ICD10 i10 ON i9.ICD9Description = i10.Description;

-- Step 4: Create the final temporary table
IF OBJECT_ID('tempdb..#Final_ICD9_ICD10_Table', 'U') IS NOT NULL DROP TABLE #Final_ICD9_ICD10_Table;

SELECT 
    i9m.ICD9code,
    i9m.ICD9Description,
    i9m.ICD10code,
    i9m.ICD10Description
INTO #Final_ICD9_ICD10_Table
FROM #ICD9_ICD10_Match i9m;

-- Verify the final table
SELECT * FROM #Final_ICD9_ICD10_Table;


-- Step 1: Create a temporary table for the new crosswalk with added codes
IF OBJECT_ID('ICD9ToICD10DxCrosswalkUpdated', 'U') IS NOT NULL DROP TABLE dbo.ICD9ToICD10DxCrosswalkUpdated;

-- Insert existing mappings from ICD9ToICD10DxCrosswalkSimple
SELECT *
INTO ICD9ToICD10DxCrosswalkUpdated
FROM ICD9ToICD10DxCrosswalkSimple;

-- Insert the additional mappings from #Final_ICD9_ICD10_Table
INSERT INTO ICD9ToICD10DxCrosswalkUpdated (ICD9, ICD10)
SELECT ICD9code AS ICD9, ICD10code AS ICD10
FROM #Final_ICD9_ICD10_Table;

-- Verify the new crosswalk table
SELECT * FROM ICD9ToICD10DxCrosswalkUpdated;


-- Step 2: Update Diagnoses_ICD10 table with ICD9 codes replaced by ICD10 codes with new crosswalk
IF OBJECT_ID('dbo.Diagnoses_ICD10', 'U') IS NOT NULL DROP TABLE dbo.Diagnoses_ICD10;

-- Replace ICD9 codes with ICD10 codes
SELECT 
    d.StudyID,
    d.Date,
    COALESCE(c.ICD10, d.Code) AS Code,
    CASE 
        WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
        ELSE d.Code_Type
    END AS Code_Type,
    d.IndexDate,
    COALESCE(c.ICD10, d.Code) + '_' + CASE 
                                         WHEN c.ICD10 IS NOT NULL THEN 'ICD10'
                                         ELSE d.Code_Type
                                       END AS CodeWithType
INTO dbo.Diagnoses_ICD10
FROM #CTE_Diagnoses d
LEFT JOIN ICD9ToICD10DxCrosswalkUpdated c
ON d.Code = c.ICD9
WHERE d.Code_Type = 'ICD9' OR d.Code_Type = 'ICD10';

-- Delete remaining ICD9 codes
DELETE FROM dbo.Diagnoses_ICD10
WHERE Code_Type = 'ICD9';





-- Step 4: Print the number of ICD9 codes remaining after using the crosswalk
-- 335441
SELECT COUNT(*) AS Remaining_ICD9_Count
FROM dbo.Diagnoses_ICD10
WHERE Code_Type = 'ICD9';

-- Step 5: List the ICD9 codes remaining after using the crosswalk
SELECT DISTINCT Code AS Remaining_ICD9_Codes
FROM dbo.Diagnoses_ICD10
WHERE Code_Type = 'ICD9';

-- Step 6: Print the number of distinct ICD9 codes remaining
-- 1875
SELECT COUNT(DISTINCT Code) AS Distinct_Remaining_ICD9_Codes
FROM dbo.Diagnoses_ICD10
WHERE Code_Type = 'ICD9';

SELECT 
    Code AS ICD9_Code, 
    COUNT(*) AS Frequency
FROM 
    dbo.Diagnoses_ICD10
WHERE 
    Code_Type = 'ICD9'
GROUP BY 
    Code
ORDER BY 
    Frequency DESC;




-- Step 1: Create a temporary table for the remaining ICD9 codes from the Diagnoses_ICD10 table
IF OBJECT_ID('tempdb..#Remaining_ICD9_Codes', 'U') IS NOT NULL DROP TABLE #Remaining_ICD9_Codes;

SELECT DISTINCT Code AS ICD9_Remaining
INTO #Remaining_ICD9_Codes
FROM dbo.Diagnoses_ICD10
WHERE Code_Type = 'ICD9';

-- Step 2: Create a temporary table for the ICD9 codes from the updated crosswalk
IF OBJECT_ID('tempdb..#Updated_Crosswalk_ICD9', 'U') IS NOT NULL DROP TABLE #Updated_Crosswalk_ICD9;

SELECT DISTINCT ICD9 AS ICD9_Crosswalk, ICD10 AS ICD10_Crosswalk
INTO #Updated_Crosswalk_ICD9
FROM ICD9ToICD10DxCrosswalkUpdated;

-- Step 3: Create a function to check for at least 4 consecutive matching digits and place value, including letter match
IF OBJECT_ID('dbo.fn_MatchICD9Codes', 'FN') IS NOT NULL DROP FUNCTION dbo.fn_MatchICD9Codes;
GO

CREATE FUNCTION dbo.fn_MatchICD9Codes (@icd9_remain VARCHAR(50), @icd9_cross VARCHAR(50))
RETURNS BIT
AS
BEGIN
    DECLARE @i INT = 1;
    DECLARE @match BIT = 0;
    DECLARE @remain_clean VARCHAR(50) = REPLACE(@icd9_remain, '.', '');
    DECLARE @cross_clean VARCHAR(50) = REPLACE(@icd9_cross, '.', '');

    WHILE @i <= LEN(@remain_clean) - 3 AND @match = 0
    BEGIN
        -- Check for letter at the beginning and 4 digits after
        IF (PATINDEX('%[A-Z]%', LEFT(@remain_clean, 1)) = 1 AND LEFT(@remain_clean, 1) = LEFT(@cross_clean, 1)
            AND SUBSTRING(@remain_clean, 2, 4) = SUBSTRING(@cross_clean, 2, 4))
            OR (SUBSTRING(@remain_clean, @i, 4) = SUBSTRING(@cross_clean, @i, 4))
        BEGIN
            SET @match = 1;
        END
        SET @i = @i + 1;
    END
    RETURN @match;
END;
GO

-- Step 4: Create a temporary table to store the matching ICD9 and ICD10 codes
IF OBJECT_ID('tempdb..#Matching_ICD9_ICD10', 'U') IS NOT NULL DROP TABLE #Matching_ICD9_ICD10;

SELECT 
    r.ICD9_Remaining,
    c.ICD9_Crosswalk,
    c.ICD10_Crosswalk
INTO #Matching_ICD9_ICD10
FROM #Remaining_ICD9_Codes r
JOIN #Updated_Crosswalk_ICD9 c
ON dbo.fn_MatchICD9Codes(REPLACE(r.ICD9_Remaining, '.', ''), REPLACE(c.ICD9_Crosswalk, '.', '')) = 1;

-- Verify the resulting table
SELECT TOP(1000)* FROM #Matching_ICD9_ICD10;





IF OBJECT_ID('tempdb..#ManualMappingsDiagnoses', 'U') IS NOT NULL DROP TABLE #ManualMappingsDiagnoses;

CREATE TABLE #ManualMappingsDiagnoses (
    ICD9 VARCHAR(10),
    ICD10 VARCHAR(10)
);

INSERT INTO #ManualMappingsDiagnoses (ICD9, ICD10)
VALUES
    ('V70.0', 'Z00.00'),
    ('V04.81', 'Z23'),
    ('V76.2', 'Z12.4'),
    ('V72.81', 'Z01.41'),
    ('V76.12', 'Z12.72'),
    ('V72.84', 'Z01.89'),
    ('V58.69', 'Z79.899'),
    ('V76.51', 'Z12.39'),
    ('V04.8', 'Z23'),
    ('V03.82', 'Z23'),
    ('V72.3', 'Z01.00'),
    ('780.6', 'R53.83'),
    ('V72.31', 'Z01.00'),
    ('V72.83', 'Z01.89'),
    ('V45.89', 'Z98.890'),
    ('V15.82', 'Z91.19'),
    ('V06.1', 'Z23'),
    ('V06.5', 'Z23'),
    ('793.1', 'R91.8'),
    ('V72.6', 'Z13.89'),
    ('V76.10', 'Z12.4'),
    ('V58.81', 'Z79.890'),
    ('V76.1', 'Z12.4'),
    ('V15.89', 'Z91.19'),
    ('V05.3', 'Z23'),
    ('V58.67', 'Z79.899'),
    ('V71.89', 'Z04.6'),
    ('V58.61', 'Z79.01'),
    ('V12.72', 'Z86.69'),
    ('795.5', 'R87.619'),
    ('789.0', 'R10.9'),
    ('V76.44', 'Z12.89'),
    ('V49.9', 'Z72.0'),
    ('V74.1', 'Z11.4'),
    ('V17.3', 'Z82.49'),
    ('V58.66', 'Z79.899'),
    ('V58.82', 'Z79.890'),
    ('V16.3', 'Z80.3'),
    ('V45.82', 'Z95.2'),
    ('V45.81', 'Z95.1'),
    ('V65.3', 'Z71.3'),
    ('V72.5', 'Z01.10'),
    ('V27.0', 'Z37.0'),
    ('V70.9', 'Z00.00'),
    ('V71.4', 'Z04.8'),
    ('V74.5', 'Z11.1'),
    ('959.1', 'S39.93XA'),
    ('600', 'N40.0'),
    ('795.0', 'R92.8'),
    ('V82.81', 'Z13.89'),
    ('V49.81', 'Z71.2'),
    ('V15.81', 'Z91.19'),
    ('V16.0', 'Z80.0'),
    ('V06.8', 'Z23'),
    ('279.4', 'D83.9'),
    ('V22.1', 'Z34.91'),
    ('600.0', 'N40.0'),
    ('V72.82', 'Z01.12'),
    ('V58.49', 'Z79.899'),
    ('V67.09', 'Z09'),
    ('414.0', 'I25.10'),
    ('V65.40', 'Z76.0'),
    ('V76.49', 'Z12.89'),
    ('V05.9', 'Z23'),
    ('V70.3', 'Z02.0'),
    ('V76.11', 'Z12.4'),
    ('V71.9', 'Z02.89'),
    ('V25.9', 'Z30.9'),
    ('622.1', 'N87.9');

INSERT INTO ICD9ToICD10DxCrosswalkUpdated (ICD9, ICD10)
SELECT ICD9, ICD10
FROM #ManualMappingsDiagnoses;




-------------------------------------------------------------------------------------------------------
-- Making Results column numeric
--------------------------------------------------------------------------------------------------------
-- refer to line 632


-- Drop existing table if exists
IF OBJECT_ID('dbo.Labs_combined', 'U') IS NOT NULL DROP TABLE dbo.Labs_combined;

-- Combined CTE for labs from EPIC, LMR, and Archive
;WITH CombinedLabs AS (
    SELECT
        e.EMPI,
        e.LabDate AS Date,
        e.ComponentCommonNM AS Code,
        TRY_CAST(e.NVal AS FLOAT) AS Result_n, -- Safe conversion to float
        e.TVal AS Result_t, -- Keep character results directly
        CASE WHEN TRY_CAST(e.NVal AS FLOAT) IS NOT NULL THEN 'Numeric' ELSE 'String' END AS ValType, -- Determine ValType
        'EPIC' AS Source
    FROM dbo.labsEpic e
    JOIN vw_matching_labs ml ON e.ComponentCommonNM = ml.ComponentCommonNM
    WHERE e.LabDate BETWEEN @minDate AND @maxDate
          AND (e.NVal IS NOT NULL OR e.TVal IS NOT NULL)
    UNION ALL
    SELECT
        l.empi,
        l.LabDate,
        l.GroupCD AS Code,
        TRY_CAST(l.NVal AS FLOAT) AS Result_n,
        l.TVal AS Result_t,
        CASE WHEN TRY_CAST(l.NVal AS FLOAT) IS NOT NULL THEN 'Numeric' ELSE 'String' END AS ValType,
        'LMR' AS Source
    FROM dbo.labsLMR l
    JOIN vw_matching_labs ml ON l.GroupCD = ml.GroupCD
    WHERE l.LabDate BETWEEN @minDate AND @maxDate
          AND (l.NVal IS NOT NULL OR l.TVal IS NOT NULL)
    UNION ALL
    SELECT
        a.EMPI,
        a.LabDate,
        a.GroupCD AS Code,
        TRY_CAST(a.nval AS FLOAT) AS Result_n,
        a.tval AS Result_t,
        CASE WHEN TRY_CAST(a.nval AS FLOAT) IS NOT NULL THEN 'Numeric' ELSE 'String' END AS ValType,
        'Archive' AS Source
    FROM dbo.labsLMRArchive a
    JOIN vw_matching_labs ml ON a.GroupCD = ml.GroupCD
    WHERE a.LabDate BETWEEN @minDate AND @maxDate
          AND (a.nval IS NOT NULL OR a.tval IS NOT NULL)
),
-- Join with index date and crosswalk
LabsWithIndexDateAndCrosswalk AS (
    SELECT
        cl.EMPI,
        x.StudyID,
        cl.Date,
        cl.Code,
        cl.Result_n,
        cl.Result_t,
        cl.ValType,
        cl.Source
    FROM CombinedLabs cl
    INNER JOIN #tmp_indexDate idx ON cl.EMPI = idx.EMPI AND cl.Date <= idx.IndexDate
    LEFT JOIN CrosswalkTable x ON cl.EMPI = x.EMPI
)

-- Insert into the final Labs_combined table
SELECT
    ISNULL(lwc.StudyID, lwc.EMPI) AS StudyID, -- Use StudyID if available, else EMPI
    lwc.Date,
    lwc.Code,
    lwc.Result_n,
    lwc.Result_t,
    lwc.ValType,
    lwc.Source
INTO dbo.Labs_combined
FROM LabsWithIndexDateAndCrosswalk lwc;

-- Output the results for verification
SELECT TOP 100 * FROM dbo.Labs_combined;








-------------------------------------------------------------------------------------------------------
-- Vitals signs phy_2_pcp_combined
--------------------------------------------------------------------------------------------------------


-- Query to get distinct Concept_Name with their counts, sorted in descending order
SELECT 
    Concept_Name,
    COUNT(*) AS Concept_Count
FROM 
    dbo.phy_2_pcp_combined
GROUP BY 
    Concept_Name
ORDER BY 
    Concept_Count DESC;



IF OBJECT_ID('dbo.phy_2_pcp_combined_StudyID', 'U') IS NOT NULL DROP TABLE dbo.phy_2_pcp_combined_StudyID;

-- Create phy_2_pcp_combined_StudyID table
SELECT
    s.StudyID,
    p.EPIC_PMRN,
    p.MRN_Type,
    p.MRN,
    p.Date,
    p.Concept_Name,
    p.Code_Type,
    p.Code,
    p.Result,
    p.Units,
    p.Provider,
    p.Clinic,
    p.Hospital,
    p.Inpatient_Outpatient,
    p.Encounter_number
INTO dbo.phy_2_pcp_combined_StudyID
FROM dbo.phy_2_pcp_combined p
LEFT JOIN dbo.StudyID s ON p.EMPI = s.EMPI;



IF OBJECT_ID('dbo.phy_2_pcp_combined_limited', 'U') IS NOT NULL DROP TABLE dbo.phy_2_pcp_combined_limited;

-- Create phy_2_pcp_combined_limited table
SELECT
    StudyID,
    Date,
    CASE 
        WHEN Concept_Name = 'Diastolic-LFA3952.2' THEN 'Diastolic-Epic'
        WHEN Concept_Name = 'Systolic-LFA3959.1' THEN 'Systolic-Epic'
        ELSE Concept_Name
    END AS Concept_Name,
    Code_Type,
    Code,
    Result,
    Units,
    Provider,
    Clinic,
    Hospital,
    Inpatient_Outpatient,
    Encounter_number
INTO dbo.phy_2_pcp_combined_limited
FROM dbo.phy_2_pcp_combined_StudyID
WHERE Concept_Name IN (
    'Pulse',
    'Weight',
    'BMI',
    'Temperature',
    'Diastolic-Epic',
    'Systolic-Epic',
    'Diastolic-LFA3952.2',
    'Systolic-LFA3959.1',
    'O2 Saturation-SPO2',
    'Respiratory rate',
    'Pain Score'
);



-------------------------------------------------------------------------------------------------------
--Medications 
--------------------------------------------------------------------------------------------------------

IF OBJECT_ID('dbo.medsEPIC_simple', 'U') IS NOT NULL DROP TABLE dbo.medsEPIC_simple;


SELECT
    s.StudyID,
    e.OrderDTS,
    e.PharmaceuticalSubclassCD,
    e.PharmaceuticalSubclassDSC,
    e.SimpleGenericDSC,
    LEFT(e.SimpleGenericDSC, CHARINDEX(' ', e.SimpleGenericDSC + ' ') - 1) AS SimpleGenericFirstWord,
    e.OrderingModeDSC
INTO dbo.medsEPIC_simple
FROM dbo.medsEPIC e
LEFT JOIN dbo.StudyID s ON e.EMPI = s.EMPI
WHERE e.OrderDTS IS NOT NULL;


SELECT TOP(1000) * 
FROM medsEPIC_simple;


SELECT 
    COUNT(*) AS Null_StudyID_Count
FROM 
    medsEPIC_simple 
WHERE 
    StudyID IS NULL;
-- 0


SELECT 
    COUNT(*) AS Null_OrderDTS_Count
FROM 
    medsEPIC_simple
WHERE 
    OrderDTS IS NULL;
-- 0




-- Step 1: Drop the existing medsLMR_StudyID table if it exists
IF OBJECT_ID('dbo.medsLMR_StudyID', 'U') IS NOT NULL DROP TABLE dbo.medsLMR_StudyID;

-- Step 2: Create the medsLMR_StudyID table with replaced EMPI and without qdrMedname and qdrMednameFirstWord
SELECT
    s.StudyID,
    l.MedFactID,
    l.recordID,
    l.RecSeq,
    l.rollupID,
    l.Medname,
    l.serviceDate,
    l.factAudit,
    l.statusCode,
    l.startdate,
    l.stopdate,
    l.dose,
    l.doseunits,
    l.route,
    l.frequency,
    l.strength,
    l.duration,
    l.durationunits,
    l.refills,
    l.prescriptionFlag,
    l.type,
    l.nosubsflag,
    l.comments,
    l.directions,
    l.prn,
    l.prnReason,
    l.take,
    l.doseType,
    l.slidingScale,
    l.provid,
    l.providercode,
    l.clinid,
    l.clinicCode,
    l.entryDate,
    l.OriginalDate,
    l.LastActionTaken,
    l.VerifyAction,
    l.dischargeReconciliationID,
    l.PrescriptionGeneratedByLMR,
    l.RetailMailPharmacy,
    l.AsDirectedflag
INTO dbo.medsLMR_StudyID
FROM dbo.medsLMR l
LEFT JOIN dbo.StudyID s ON l.EMPI = s.EMPI
LEFT JOIN dbo.qdrMedications q ON l.rollupID = q.rollupID
WHERE q.route <> 'NONMED' OR q.route IS NULL;

-- Step 3: Output the results for verification
SELECT TOP 1000 * FROM dbo.medsLMR_StudyID;



-- Check for discrepancies in StudyID assignment between medsEPIC_simple and medsLMR_StudyID for the same EMPI

WITH EMPI_to_StudyID_EPIC AS (
    SELECT DISTINCT e.EMPI, s.StudyID AS StudyID_EPIC
    FROM dbo.medsEPIC e
    LEFT JOIN dbo.StudyID s ON e.EMPI = s.EMPI
),
EMPI_to_StudyID_LMR AS (
    SELECT DISTINCT l.EMPI, s.StudyID AS StudyID_LMR
    FROM dbo.medsLMR l
    LEFT JOIN dbo.StudyID s ON l.EMPI = s.EMPI
)
SELECT 
    e.EMPI,
    e.StudyID_EPIC,
    l.StudyID_LMR
INTO #DiscrepancyCheck -- Save results into a temporary table
FROM EMPI_to_StudyID_EPIC e
JOIN EMPI_to_StudyID_LMR l ON e.EMPI = l.EMPI
WHERE e.StudyID_EPIC <> l.StudyID_LMR;

-- Count the number of discrepancies found
SELECT COUNT(*) AS Discrepancies
FROM #DiscrepancyCheck; -- Use the temporary table to count discrepancies

-- 0 discrepancies






-- Drop the meds_combined table if it exists to recreate it
IF OBJECT_ID('dbo.meds_combined', 'U') IS NOT NULL DROP TABLE dbo.meds_combined;

-- Create the meds_combined table combining medsLMR_StudyID and medsEPIC_simple
WITH CombinedMeds AS (
    SELECT 
        l.StudyID,
        l.serviceDate AS Order_Date,  -- Use serviceDate instead of startdate
        l.Medname AS Medication_Name,
        'LMR' AS Source
    FROM dbo.medsLMR_StudyID l
    WHERE l.Medname IS NOT NULL  -- Exclude rows where Medname is NULL

    UNION ALL

    SELECT 
        e.StudyID,
        e.OrderDTS AS Order_Date,
        e.SimpleGenericDSC AS Medication_Name,
        'EPIC' AS Source
    FROM dbo.medsEPIC_simple e
    WHERE e.SimpleGenericDSC IS NOT NULL  -- Exclude rows where SimpleGenericDSC is NULL
)
, DeduplicatedMeds AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY StudyID, Medication_Name, Order_Date, Source ORDER BY StudyID) AS RowNum
    FROM CombinedMeds
)

-- Select into meds_combined, removing duplicates
SELECT 
    StudyID,
    Medication_Name,
    Order_Date,
    Source,
    LEFT(Medication_Name, CHARINDEX(' ', Medication_Name + ' ') - 1) AS medicationNameFirstWord  -- Extract first word of Medication_Name
INTO dbo.meds_combined
FROM DeduplicatedMeds
WHERE RowNum = 1;


-- Output the top 1000 rows for verification
SELECT TOP 1000 * FROM dbo.meds_combined;



---- mednameFirstWord no longer part of table, ignore this part
---- Step 1: Count the number of unique mednameFirstWord found in simpleGenericFirstWord in medsEpic_simple
--SELECT 
--    COUNT(DISTINCT lmr.mednameFirstWord) AS FoundInEpic
--FROM 
--    dbo.medsLMR_StudyID lmr
--INNER JOIN 
--    dbo.medsEpic_simple epic
--ON 
--    lmr.mednameFirstWord = epic.simpleGenericFirstWord;




---- Step 2: Count the number of unique mednameFirstWord not found in simpleGenericFirstWord in medsEpic_simple
--SELECT 
--    COUNT(DISTINCT lmr.mednameFirstWord) AS NotFoundInEpic
--FROM 
--    dbo.medsLMR_StudyID lmr
--LEFT JOIN 
--    dbo.medsEpic_simple epic
--ON 
--    lmr.mednameFirstWord = epic.simpleGenericFirstWord
--WHERE 
--    epic.simpleGenericFirstWord IS NULL;




---- Step 3: Optional - View the list of unique mednameFirstWord not found in simpleGenericFirstWord in medsEpic_simple
--SELECT 
--    DISTINCT lmr.mednameFirstWord
--FROM 
--    dbo.medsLMR_StudyID lmr
--LEFT JOIN 
--    dbo.medsEpic_simple epic
--ON 
--    lmr.mednameFirstWord = epic.simpleGenericFirstWord
--WHERE 
--    epic.simpleGenericFirstWord IS NULL
--ORDER BY 
--    lmr.mednameFirstWord;





---- Create the list of mednameFirstWord found in simpleGenericFirstWord
--SELECT DISTINCT lmr.mednameFirstWord
--FROM dbo.medsLMR_StudyID lmr
--INNER JOIN dbo.medsEpic_simple epic
--ON lmr.mednameFirstWord = epic.simpleGenericFirstWord
--ORDER BY lmr.mednameFirstWord;



---- Create the list of mednameFirstWord found in simpleGenericFirstWord
--SELECT DISTINCT lmr.mednameFirstWord
--FROM dbo.medsLMR_StudyID lmr
--INNER JOIN dbo.medsEpic_simple epic
--ON lmr.mednameFirstWord = epic.simpleGenericFirstWord
--ORDER BY lmr.mednameFirstWord;




--IF OBJECT_ID('dbo.MednameFirstWord_Found', 'U') IS NOT NULL DROP TABLE dbo.MednameFirstWord_Found;

---- Create the table by selecting from the list you already have
--SELECT DISTINCT mednameFirstWord
--INTO dbo.MednameFirstWord_Found
--FROM dbo.medsLMR_StudyID l
--WHERE EXISTS (
--    SELECT 1 
--    FROM dbo.medsEpic_simple e
--    WHERE e.simpleGenericFirstWord = l.mednameFirstWord
--);




--IF OBJECT_ID('dbo.MednameFirstWord_NotFound', 'U') IS NOT NULL DROP TABLE dbo.MednameFirstWord_NotFound;

---- Create the table by selecting from the list you already have
--SELECT DISTINCT mednameFirstWord
--INTO dbo.MednameFirstWord_NotFound
--FROM dbo.medsLMR_StudyID l
--WHERE NOT EXISTS (
--    SELECT 1 
--    FROM dbo.medsEpic_simple e
--    WHERE e.simpleGenericFirstWord = l.mednameFirstWord
--);



---- Drop the table if it exists
--IF OBJECT_ID('dbo.MednameFirstWord_Combined', 'U') IS NOT NULL DROP TABLE dbo.MednameFirstWord_Combined;

---- Perform a full outer join to combine the results
--SELECT 
--    COALESCE(f.mednameFirstWord, n.mednameFirstWord) AS mednameFirstWord,
--    CASE 
--        WHEN f.mednameFirstWord IS NOT NULL THEN 1 
--        ELSE 0 
--    END AS Found,
--    CASE 
--        WHEN n.mednameFirstWord IS NOT NULL THEN 1 
--        ELSE 0 
--    END AS NotFound
--INTO dbo.MednameFirstWord_Combined
--FROM dbo.MednameFirstWord_Found f
--FULL OUTER JOIN dbo.MednameFirstWord_NotFound n 
--    ON f.mednameFirstWord = n.mednameFirstWord;

---- Output the results for verification
--SELECT TOP 2104 * FROM dbo.MednameFirstWord_Combined;


-------------------------------------------------------------------------------------------------------
-- Adding EMPIs not in StudyID crosswalk from meds data 
--------------------------------------------------------------------------------------------------------


-- Identify records with NULL StudyID in medsEPIC_simple
SELECT * 
FROM medsEPIC_simple
WHERE StudyID IS NULL;

-- Count records with NULL StudyID
SELECT COUNT(*) AS NullStudyIDCount
FROM medsEPIC_simple
WHERE StudyID IS NULL;


-- Identify unique EMPIs in medsEPIC_simple that are not in the StudyID table
SELECT DISTINCT m.EMPI
FROM medsEPIC m
LEFT JOIN StudyID s ON m.EMPI = s.EMPI
WHERE s.StudyID IS NULL;







-- Check the number of unique EMPIs in each table
SELECT 'Labs_EMPI' AS TableName, COUNT(DISTINCT EMPI) AS UniqueEMPIs FROM Labs_EMPI
UNION ALL
SELECT 'Procedures_EMPI', COUNT(DISTINCT EMPI) FROM Procedures_EMPI
UNION ALL
SELECT 'Diagnoses_EMPI', COUNT(DISTINCT EMPI) FROM Diagnoses_EMPI
UNION ALL
SELECT 'DiabetesOutcomes_EMPI', COUNT(DISTINCT EMPI) FROM DiabetesOutcomes_EMPI
UNION ALL
SELECT 'medsLMR', COUNT(DISTINCT EMPI) FROM medsLMR
UNION ALL
SELECT 'medsEPIC', COUNT(DISTINCT EMPI) FROM medsEPIC;

-- Table  UniqueEMPIs
-- Labs_EMPI    114300
-- Procedures_EMPI  53664
-- Diagnoses_EMPI   54154
-- DiabetesOutcomes_EMPI    55657
-- medsLMR  99514
-- medsEPIC 21484


-- Create temporary table to store missing EMPIs
IF OBJECT_ID('tempdb..#MissingEMPIs', 'U') IS NOT NULL DROP TABLE #MissingEMPIs;

CREATE TABLE #MissingEMPIs (
    EMPI VARCHAR(20),
    SourceTable VARCHAR(50)
);

-- Insert EMPIs not in StudyID crosswalk
INSERT INTO #MissingEMPIs (EMPI, SourceTable)
SELECT DISTINCT e.EMPI, 'Labs_EMPI'
FROM Labs_EMPI e
LEFT JOIN StudyID s ON e.EMPI = s.EMPI
WHERE s.EMPI IS NULL

UNION ALL

SELECT DISTINCT e.EMPI, 'Procedures_EMPI'
FROM Procedures_EMPI e
LEFT JOIN StudyID s ON e.EMPI = s.EMPI
WHERE s.EMPI IS NULL

UNION ALL

SELECT DISTINCT e.EMPI, 'Diagnoses_EMPI'
FROM Diagnoses_EMPI e
LEFT JOIN StudyID s ON e.EMPI = s.EMPI
WHERE s.EMPI IS NULL

UNION ALL

SELECT DISTINCT e.EMPI, 'DiabetesOutcomes_EMPI'
FROM DiabetesOutcomes_EMPI e
LEFT JOIN StudyID s ON e.EMPI = s.EMPI
WHERE s.EMPI IS NULL

UNION ALL

SELECT DISTINCT e.EMPI, 'medsLMR'
FROM medsLMR e
LEFT JOIN StudyID s ON e.EMPI = s.EMPI
WHERE s.EMPI IS NULL

UNION ALL

SELECT DISTINCT e.EMPI, 'medsEPIC'
FROM medsEPIC e
LEFT JOIN StudyID s ON e.EMPI = s.EMPI
WHERE s.EMPI IS NULL;

-- Check how many unique EMPIs are missing from each table
SELECT SourceTable, COUNT(DISTINCT EMPI) AS MissingEMPIs
FROM #MissingEMPIs
GROUP BY SourceTable;


-- Source Table  MissingEMPIs
-- medsEPIC 7600
-- medsLMR  1429


-- Add missing EMPIs to the StudyID crosswalk
INSERT INTO StudyID (EMPI, StudyID)
SELECT DISTINCT EMPI, NEWID()
FROM #MissingEMPIs;

-- Step 1: Create a temporary table to hold unique EMPIs
IF OBJECT_ID('tempdb..#UniqueMissingEMPIs', 'U') IS NOT NULL DROP TABLE #UniqueMissingEMPIs;

SELECT DISTINCT EMPI 
INTO #UniqueMissingEMPIs
FROM #MissingEMPIs;

-- Step 2: Insert unique EMPIs into StudyID with NEWID() for each missing EMPI
INSERT INTO StudyID (EMPI, StudyID)
SELECT u.EMPI, NEWID()
FROM #UniqueMissingEMPIs u
LEFT JOIN StudyID s ON u.EMPI = s.EMPI
WHERE s.EMPI IS NULL;