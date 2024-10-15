def calculate_percentage(result_df):
    total_values = result_df.size  # Total number of values in the DataFrame
    true_count = (result_df == True).sum().sum()  # Count of True values
    false_count = (result_df == False).sum().sum()  # Count of False values
    
    # Calculate percentages
    true_percentage = (true_count / total_values) * 100
    false_percentage = (false_count / total_values) * 100
    
    return {
        'True_Percentage': true_percentage,
        'False_Percentage': false_percentage
    }

# Calculate the percentage of True and False values
percentages = calculate_percentage(result)
print(percentages)
