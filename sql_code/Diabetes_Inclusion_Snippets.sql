--Modified from Q:\Projects\DiabetesPredictiveModeling\SQL_examples\DataProcessingExamples\SnippetsforInclusionCriteria.sql

--Identify Patients with At Least Two Encounters
--select those with at least two encounters, at least one of which occurs after 1/1/00 and before 11/30/2021

IF OBJECT_ID('tempdb..#tmp_PrimCarePatients') IS NOT NULL
    DROP TABLE #tmp_PrimCarePatients;

--Identify Patients with At Least Two Encounters
IF OBJECT_ID('tempdb..#tmp_PrimCarePatients') IS NOT NULL
    DROP TABLE #tmp_PrimCarePatients;

DECLARE @minDate DATE = '2000-01-01';
DECLARE @maxDate DATE = '2022-06-30';

SELECT a.EMPI, 
       COUNT(DISTINCT a.serviceDate) AS NumberEncounters, 
       MIN(a.serviceDate) AS minDate, 
       MAX(a.serviceDate) AS maxDate, 
       DATEADD(YEAR, 18, d.Date_of_Birth) AS Date18, 
       d.Date_of_Birth, 
       d.Gender_Legal_Sex
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
HAVING COUNT(DISTINCT a.serviceDate) > 1;

--Clean A1c Values
DECLARE @min_A1c FLOAT = 4.0;

IF OBJECT_ID('CleanedA1cAverages') IS NOT NULL
    DROP TABLE CleanedA1cAverages;

SELECT EMPI, d, AVG(A1c) AS A1c
INTO CleanedA1cAverages
FROM (
    SELECT EMPI, CAST(LabDate AS DATE) AS d, nval AS 'A1c'
    FROM labsEPIC
    WHERE StudyLabCode = 'HbA1c' 
          AND ISNUMERIC(nval) > 0 
          AND nval >= @min_A1c
) AS A 
GROUP BY A.EMPI, d;

--Windows of Elevated A1c
DECLARE @Study_Start DATE = '2000-01-01';
DECLARE @Study_End DATE = '2022-12-31';
DECLARE @min_A1c_inclusion FLOAT = 7.0; 
DECLARE @FollowUpMonths INT = 12;

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
);

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
WHERE b.LabDate > @minDate 
  AND b.GroupCD = 'HbA1c' 
  AND TRY_CONVERT(FLOAT, b.nval) >= @min_A1c_inclusion;

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
WHERE a.LabDate > @minDate 
  AND a.GroupCD = 'HbA1c' 
  AND TRY_CONVERT(FLOAT, a.nval) >= @min_A1c_inclusion;

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
WHERE e.LabDate > @minDate 
  AND e.StudyLabCode = 'HbA1c' 
  AND TRY_CONVERT(FLOAT, e.nval) >= @min_A1c_inclusion;

-- Identify Patients with Diabetes at Baseline
DECLARE @Baseline_End DATE = '2022-06-30';

-- Check if #tmp_DMpatsBaseline already exists and drop it if it does
IF OBJECT_ID('tempdb..#tmp_DMpatsBaseline') IS NOT NULL
    DROP TABLE #tmp_DMpatsBaseline;

-- Create #tmp_DMpatsBaseline
SELECT DISTINCT CASE 
    WHEN d.EMPI IS NOT NULL THEN d.EMPI
    WHEN d.EMPI IS NULL THEN e.EMPI
    END AS EMPI, 
    d.firstInptDMDate, 
    e.firstOutptDMDate,
    CASE 
        WHEN d.firstInptDMDate <= e.firstOutptDMDate OR e.firstOutptDMDate IS NULL THEN d.firstInptDMDate
        WHEN d.firstInptDMDate > e.firstOutptDMDate OR d.firstInptDMDate IS NULL THEN e.firstOutptDMDate
    END AS firstDMDateBase, 
    d.countInptDM, 
    e.countOutptDM, 
    DMBaseline = '1'
INTO #tmp_DMpatsBaseline
FROM (
    SELECT DISTINCT dx.EMPI, COUNT(DISTINCT dx.Date) AS countInptDM, MIN(CONVERT(DATE, dx.Date)) AS firstInptDMDate
    FROM dia_2_pcp_combined dx
    WHERE (dx.Code LIKE '250.%' OR dx.Code LIKE 'E10%' OR dx.Code LIKE 'E11%')
          AND CONVERT(DATE, dx.Date) < @Baseline_End
    GROUP BY dx.EMPI
) AS d
FULL JOIN (
    SELECT DISTINCT dx.EMPI, COUNT(DISTINCT dx.Date) AS countOutptDM, MIN(CONVERT(DATE, dx.Date)) AS firstOutptDMDate
    FROM dia_2_pcp_combined dx
    WHERE (dx.Code LIKE '250.%' OR dx.Code LIKE 'E10%' OR dx.Code LIKE 'E11%')
          AND CONVERT(DATE, dx.Date) < @Baseline_End
    GROUP BY dx.EMPI
) AS e ON d.EMPI = e.EMPI
WHERE d.countInptDM > 0 OR e.countOutptDM > 1;

-- Determine Eligible Periods and Index Date
SELECT p.EMPI, MIN(a.A1cDate) AS IndexDate
INTO #tmp_indexDate
FROM EMPIs2PrimaryCareEnc02 p
INNER JOIN #tmp_A1cElevated a ON a.EMPI = p.EMPI
INNER JOIN dem_2_pcp_combined d ON p.EMPI = d.EMPI
WHERE a.A1cDate BETWEEN p.FirstEncounterDate AND p.LastEncounterDate
      AND TRY_CONVERT(FLOAT, a.nval) >= @min_A1c_inclusion
      AND DATEDIFF(year, d.Date_of_Birth, a.A1cDate) BETWEEN 18 AND 75
      AND (d.Date_of_Death IS NULL OR DATEDIFF(month, a.A1cDate, TRY_CONVERT(DATE, d.Date_of_Death, 101)) > @FollowUpMonths)
      AND a.A1cDate BETWEEN @Study_Start AND @Study_End
GROUP BY p.EMPI;
