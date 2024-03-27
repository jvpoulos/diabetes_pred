outcomes_columns = {
'studyID': 'int32',
'EMPI': 'int32',
'IndexDate': 'object',
'InitialA1c': 'float32',
'A1cAfter12Months': 'float32',
'A1cGreaterThan7': 'int32',
'Female': 'int32',
'Married': 'int32',
'GovIns': 'int32',
'English': 'int32',
'DaysFromIndexToInitialA1cDate': 'int32',
'DaysFromIndexToA1cDateAfter12Months': 'int32',
'DaysFromIndexToFirstEncounterDate': 'int32',
'DaysFromIndexToLastEncounterDate': 'int32',
'DaysFromIndexToLatestDate': 'int32',
'DaysFromIndexToPatientTurns18': 'int32',
'AgeYears': 'int32',
'BirthYear': 'int32',
'NumberEncounters': 'int32',
'SDI_score': 'float32',
'Veteran': 'int32'
}

dia_columns = {
'EMPI': 'int32',
'Date': 'object', 
'Code_Type': 'object', 
'Code': 'object', 
'IndexDate': 'object', 
'DiagnosisBeforeOrOnIndexDate': 'int32', 
'CodeWithType': 'object'
}

prc_columns = {
'EMPI': 'int32',
'Date': 'object', 
'Code_Type': 'object', 
'Code': 'object', 
'IndexDate': 'object', 
'ProcedureBeforeOrOnIndexDate': 'int32', 
'CodeWithType': 'object'
}

# Select columns to read in each dataset

outcomes_columns_select = ['studyID','EMPI', 'InitialA1c', 'A1cAfter12Months', 'A1cGreaterThan7', 'Female', 'Married', 'GovIns', 'English','AgeYears', 'BirthYear', 'SDI_score', 'Veteran']

dia_columns_select = ['EMPI', 'DiagnosisBeforeOrOnIndexDate', 'CodeWithType']

prc_columns_select = ['EMPI', 'ProcedureBeforeOrOnIndexDate', 'CodeWithType']