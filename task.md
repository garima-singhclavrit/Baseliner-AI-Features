### Need to take this from json -> For models.py
 
API endpoints:
### from 1st java api {jira features}
Task name
Task description
Task Label
Original Estimate (Planned)
Sprint number
Task priority

get_api>>
 
### from 2nd java api {et tool user features}
Optimistic Estimate
Most Likely Estimate
Pessimistic Estimate
 
### Output to java api
AI Estimate
 
  
### Need to calculate -> Use formula from jupyter notebook (cell 43 )
<!-- Note multiply weighted avergae by 100 -->
1. Weighted Average
2. Standard Deviation
3. Risk Factor
4. Complexity Factor (Variance)
5. Confidence Interval(Z-score) {Standard, High, Higher, Highest}
6. Final Estimate (3-points)
 
 

### Use cells in notebook to calculate encodings (Cell no 50) (use gpt model cell 20,21,22) 
### load model 
### perform prediction (cell 57)
### do post processing