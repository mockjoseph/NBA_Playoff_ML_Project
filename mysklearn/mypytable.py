from mysklearn import myutils
import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """
    Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        m_col = 0
        n_row = 0
        for items in self.column_names:
            m_col += 1

        for rows in self.data:
            n_row += 1

        # Should now be implemented , need to test if the for loops wrk Q: Is there an easier way to do this with a size getter?

        return n_row, m_col # TODO: test

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        # Below should work if we are only using columns that include missing values
        # will need to implement a separate function or case where include missing values is false
        col_extract = []

        index = self.column_names.index(col_identifier)

        for rows in self.data:
            col_extract.append(rows[index])

            


        return col_extract # TODO: test

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """

        row_index = 0
        col_index = 0
        for rows in self.data:
            col_index = 0
            for items in rows:
                try:
                    self.data[row_index][col_index] = float(items)
                    col_index += 1

                except (ValueError, TypeError):
                    col_index += 1
                    pass

            row_index += 1

        return self # TODO: test

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        # Make a copy
        new_table = []
        row_index = 0
        for rows in self.data:
            if row_index in row_indexes_to_drop:
                row_index += 1
            else:
                new_table.append(rows)
                row_index += 1
            
        self.data = new_table
        
        return self

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        
        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, "r") as myFile:
            the_reader = csv.reader(myFile)
            
            i = 0
            for row in the_reader:
                if len(row) > 0:
                    if i == 0:
                        i += 1
                        for items in row:
                            self.column_names.append(items)
                    else:
                        self.data.append(row)

                    
        self.convert_to_numeric()
        #print(self.column_names)
        #for rows in self.data:
        #    print(rows)

        # TODO: test
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """

        with open(filename, "w") as myFile:
            the_writer = csv.writer(myFile)
            the_writer.writerow(self.column_names)
            for row in self.data:
                the_writer.writerow(row)


        pass # TODO: fix this

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        # Get the column indexes we will be looking at
        index_list = []
        for items in self.column_names:
            if items in key_column_names:
                index_list.append(self.column_names.index(items))
        # Index list should now be created

        duplicate_list = []
        seen_values = []
        tuples = []
        for row_index, row in enumerate(self.data):
            seen_values = []
            for col_index, items in enumerate(row):
                if col_index in index_list:
                    seen_values.append(items)

            seen_values = tuple(seen_values)
            if seen_values in tuples:
                duplicate_list.append(row_index)
            else:
                tuples.append(seen_values)
                    



        return duplicate_list # TODO: fix this

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        new_table = []
        
        for row in self.data:
            add_row = True
            for items in row:
                if items == "" or items == "NA":
                    add_row = False

            if add_row == True:
                new_table.append(row)
                    
        
        self.data = new_table
        pass # TODO: fix this

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        
        col_index = self.column_names.index(col_name)
        # Get average
        total = 0
        count = 0
        for row in self.data:
            for i, item in enumerate(row):
                if i == col_index and item != "" and item != 'NA':
                    try:
                        total += float(item)
                        count += 1
                    except (ValueError, TypeError):
                        continue

        average = total / count
        #print(f"average : {average}")
        # input data
        for row in self.data:
            for i, items in enumerate(row):
                if i == col_index:
                    try:
                        items = float(items)
                    except (ValueError, TypeError):
                        row[i] = average
                    


        pass # TODO: fix this

   

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        # create a list of indexes to look at
        new_table = []
        index_list = []
        for items in self.column_names:
            if items in col_names:
                index_list.append(self.column_names.index(items))

        
        for index in index_list:
            vals = []
            for rows in self.data:
                try:
                    num = float(rows[index])
                except (ValueError, TypeError):
                    continue
                vals.append(num)
            if len(vals) != 0:
                min_val = min(vals)
                max_val = max(vals)
                mid = (max_val + min_val) / 2
                avg = sum(vals) / len(vals)
                median = 0
                vals.sort()
            
                if len(vals) % 2 == 1:
                    median = vals[len(vals) // 2]
                else:
                    median1 = vals[(len(vals) // 2) - 1]
                    median2 = vals[(len(vals) // 2)]
                    median = (median1 + median2) / 2
                vals_to_add = [self.column_names[index], min_val, max_val, mid, avg, median]
                new_table.append(vals_to_add)

        
        new_header = ["attribute", "min", "max", "mid", "avg", "median"]
        

        return MyPyTable(new_header, new_table) # TODO: fix this

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        
   
        new_header = []
        new_table = []
        for items in self.column_names:
            new_header.append(items)
        for items in other_table.column_names:
            if items not in new_header:
                new_header.append(items)

        for i, row1 in enumerate(self.data):
            for row2 in other_table.data:
                row_add = True
                for keys in key_column_names:
                    one = self.column_names.index(keys)
                    two = other_table.column_names.index(keys)
                    if row1[one] != row2[two]:
                        row_add = False
                        break
                
                if row_add == True:
                    joined_row = []

                    for i, items in enumerate(self.column_names):
                        joined_row.append(row1[i])
                        #if items not in key_column_names:
                            

                    for i, items in enumerate(other_table.column_names):
                        if items not in key_column_names:
                            joined_row.append(row2[i])

                    new_table.append(joined_row)






        return MyPyTable(new_header, new_table) # TODO: fix this


    def perform_full_outer_join(self, other_table, key_column_names):
        """
        Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """

        # Build new header: keys + left non-keys + right non-keys
        new_header = []
        for items in self.column_names:
            new_header.append(items)
        for items in other_table.column_names:
            if items not in key_column_names:
                new_header.append(items)
        #print(new_header)
        new_table = []
        matched1 = []
        matched2 = []
        for i, row1 in enumerate(self.data):
            for j, row2 in enumerate(other_table.data):
                found_match = True
                for keys in key_column_names:
                    if row1[self.column_names.index(keys)] != row2[other_table.column_names.index(keys)]:
                        found_match = False
                        break
                # If there is a match, we only want one row for it
                if found_match:
                    # Matching row pair that will already be in the table
                    matched1.append(i)
                    matched2.append(j)
                    joined_row = []
                    for c, items in enumerate(self.column_names):
                        joined_row.append(row1[c])
                    for c2, items in enumerate(other_table.column_names):
                        if items not in key_column_names:
                            joined_row.append(row2[c2])
                    new_table.append(joined_row)
                        
        # We should now have all the rows that match in the table
        # Need to add the rows that don't have matched from each table
        for i, rows in enumerate(self.data):
            joined_row = []
            if i not in matched1:
                for c, items in enumerate(self.column_names):
                    joined_row.append(rows[c])
                for items in other_table.column_names:
                    if items not in key_column_names:
                        joined_row.append("NA")
                new_table.append(joined_row)

        for j, rows in enumerate(other_table.data):
            joined_row = []
            if j not in matched2:
                for items in self.column_names:
                    joined_row.append("NA")
                for keys in key_column_names:
                    indx = self.column_names.index(keys)
                    joined_row[indx] = rows[other_table.column_names.index(keys)]
                for c, items in enumerate(other_table.column_names):
                    if items not in key_column_names:
                        joined_row.append(rows[c])
                new_table.append(joined_row)
                    
        return MyPyTable(new_header, new_table)

    def get_rows(self, indices):
        return_rows = []
        for i, row in enumerate(self.data):
            if i in indices:
                return_rows.append(row)
            


        return return_rows
    
    def drop_column(self, col_names):
        '''
        Arguements: self to change the myppytable object
        col_names : list of strings correlating to the column names to drop


        returns self
        '''
        cols_to_drop_idx = []
        new_col_names = []
        #col_check =[]
        for i, col in enumerate(self.column_names):
            if col in col_names:
                cols_to_drop_idx.append(i)
            else:
                new_col_names.append(col)
                
        self.column_names = new_col_names
        #if len(cols_to_drop_idx) != len(col_names):
         #   print("Did not recognize some column")



        new_table = []
        for row in self.data:
            new_row = []
            for i, item in enumerate(row):
                if i not in cols_to_drop_idx:
                    new_row.append(item)
            new_table.append(new_row)

        self.data = new_table

        return self

    def get_totals(self, team_name):
        n_cols = len(self.column_names)
        sums = [team_name] + [0] * n_cols
        #print(sums)
        totals = []
        for row in self.data:
            for i, val in enumerate(row):
                if i == 0:
                    if val == 'W':
                        sums[1] += 1
                else:
                    sums[i + 1] += val
                
        totals.append(sums)
        self.data = totals
        self.column_names[0] = "Wins"
        new_col_names = []
        for i in range(len(self.column_names) + 1):
            if i == 0:
                new_col_names.append("Team")
            else:
                new_col_names.append(self.column_names[i - 1])
        self.column_names = new_col_names
        return self

    def add_keyid_column(self):
        self.column_names.append('id')
        for i, row in enumerate(self.data):
            row.append(i)

    def save_DATA_to_file(self, filename):
        with open(filename, "a", newline="") as myFile:
            the_writer = csv.writer(myFile)
            for row in self.data:
                the_writer.writerow(row)

