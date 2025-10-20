import csv
import sys

csv.field_size_limit(10000000)  # Increase limit for large fields

def view_head(csv_file, column='all', num_rows=5):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= num_rows:
                break
            if column == 'id':
                print(row[0])
            elif column == 'content':
                content = row[1][:500] + '...' if len(row[1]) > 500 else row[1]
                print(content)
            elif column == 'summary':
                summary = row[2][:500] + '...' if len(row[2]) > 500 else row[2]
                print(summary)
            else:  # all
                truncated_row = [row[0], row[1][:200] + '...' if len(row[1]) > 200 else row[1], row[2][:200] + '...' if len(row[2]) > 200 else row[2]]
                print(','.join(truncated_row))

if __name__ == '__main__':
    column = sys.argv[1] if len(sys.argv) > 1 else 'all'
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    view_head('duc2006.csv', column, num_rows)