### **Documentation for `CurrentDonorForecast`**

This documentation and pseudo-code provide a roadmap for understanding and extending the `CurrentDonorForecast` VBA macro. 
For detailed implementation, ensure the structure of your Excel workbook matches the references in the macro.

This VBA macro is a forecasting tool designed for donor data analysis. 
It performs operations such as retrieving historical donor data, analysing age distributions, calculating receipt-to-demand ratios, and running Monte Carlo simulations to project future donor behaviour.

#### **Key Functionalities:**

1. **Variable Initialisation:**
   - Variables are initialised to hold data about years, daily receipts, age distributions, demand, and other parameters required for donor forecasting.

2. **User Inputs:**
   - Users are prompted to input the historical start and end years, as well as parameters for the Monte Carlo simulation, including minimum and maximum daily receipt volumes and the number of iterations.

3. **Data Extraction and Processing:**
   - Historical donor and receipt data are retrieved from worksheets.
   - Ratios such as Ratio of Receipts To People demand is calculated for each year.

4. **Monte Carlo Simulation:**
   - Simulates multiple iterations of donor data by randomly sampling daily receipt values and distributing estimated donors across age groups.

5. **Results Compilation:**
   - Outputs simulation results to worksheets for further analysis, including age-specific donor distributions for the current year.

6. **Visualisation:**
   - Summarised results, such as average ratios and means of Monte Carlo results, are saved in designated worksheets for easy interpretation.

---

### **Pseudo Code for `CurrentDonorForecast`**

```plaintext
Procedure CurrentDonorForecast:
    Initialise variables for data storage and computation.
    Set worksheet references for input data and output results.

    Prompt user for historical start and end years.
    If input is invalid, set default values.

    Prompt user for Monte Carlo parameters:
        - Minimum daily receipt
        - Maximum daily receipt
        - Number of iterations

    Retrieve historical donor data:
        - Loop through years to compute yearly receipts, demands, and receipt-to-demand ratios.
        - Store computed data in a collection.

    Process age distribution:
        - For each year:
            - Compute normalised age distributions using demand data.
            - Write results to the "AgeDistribution" worksheet.
        - For the current year:
            - Calculate average normalised age distribution.

    Perform Monte Carlo simulation:
        - For each iteration:
            - Sample a random daily receipt between Minimum and Maximum daily receipts.
            - Estimate total donors based on the sampled receipt and average ratios.
            - Distribute donors across age groups using normalised age distributions.
            - Store results in the "MonteCarloResults" worksheet.

    Calculate mean results for age groups:
        - Aggregate results across all iterations.
        - Write mean results to the "AgeSpecificDonorsCurrent" worksheet.

    Display completion messages to the user.

Function GetYearlyReceipt(ws, year):
    Search worksheet for the specified year.
    If found, return the receipt value for that year.
    Else, return 0.

End Procedure.
```

---

### **Core Functional Components**

#### **1. Input Handling:**
- **Purpose:** Gather parameters for the forecasting period and Monte Carlo simulation.
- **Techniques:** 
  - Use `InputBox` to capture user input.
  - Validate inputs and set defaults if necessary.

#### **2. Data Processing:**
- **Yearly Calculations:** Aggregate yearly receipts and demands, calculate averages and ratios.
- **Age Distribution:** Normalize age-specific donor demands for both historical and current data.

#### **3. Monte Carlo Simulation:**
- **Process:**
  - Sample random values within a defined range for daily receipts.
  - Calculate estimated donors based on sampled receipts and ratios.
  - Distribute donors across age groups using normalized demand.

#### **4. Output Results:**
- Write results to structured worksheets:
  - **`AgeDistribution`**: Contains historical and normalised age distributions.
  - **`MonteCarloResults`**: Stores simulation iteration results.
  - **`AgeSpecificDonorsCurrent`**: Provides mean donor estimates for each age group.

#### **5. Helper Function:**
- `GetYearlyReceipt`: Retrieves receipt data for a given year from a worksheet.

---