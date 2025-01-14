import streamlit as st
import six
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import base64
from io import BytesIO
from collections import defaultdict
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

class TimetableProcessor:
    def __init__(self, course_name):
        self.course_name = course_name
        self.main_header_1 = None
        self.main_header_2 = None
        self.df = None
        self.course_mappings = {}
        self.num_rows = None
        self.num_cols = None

    def extract_div_content(self, page_content_div):
        """Extract content from the specific course div."""
        if not page_content_div:
            raise ValueError("No div with class 'page_content' found.")
            
        target_div_id = f"prog_{self.course_name}"
        course_div = page_content_div.find(id=target_div_id)
        
        if not course_div:
            raise ValueError(f"Div with id '{target_div_id}' not found.")
            
        return course_div.text.strip()

    def extract_timetable_data(self, timetable):
        """Extract timetable data excluding optional courses section."""
        lines = timetable.splitlines()
        optional_course_index = next(
            (i for i, line in enumerate(lines) if line.strip() == "+ Optional course."), 
            None
        )
        
        return "\n".join(lines[:optional_course_index]) if optional_course_index else timetable

    def process_timetable(self, extracted_data):
        """Process the extracted timetable data into a DataFrame and initialize mappings."""
        lines = extracted_data.strip().splitlines()
        
        # Extract headers
        self.main_header_1 = lines[0]
        self.main_header_2 = lines[2]
        
        # Extract timings and data
        header_line = lines[5]
        data_lines = lines[7:-1]
        
        # Process columns and rows
        columns = [cell.strip() for cell in header_line.split('|') if cell.strip()]
        rows = []
        for line in data_lines:
            cells = [cell.strip() for cell in line.split('|')][:-1]
            # Capitalize the day (first column)
            cells[0] = cells[0].upper()
            rows.append(cells)
        
        # Create DataFrame
        self.df = pd.DataFrame(rows, columns=columns)
        
        # Initialize dimensions and course mappings
        self.num_rows = len(self.df.index)
        self.num_cols = len(self.df.columns)
        self._initialize_course_mappings()

    def _initialize_course_mappings(self):
        """Extract existing courses from the timetable into mappings using indices"""
        for i in range(self.num_rows):
            for j in range(1, self.num_cols):  # Skip first column (days)
                course = self.df.iloc[i, j]
                if pd.notna(course) and course.strip():
                    if course not in self.course_mappings:
                        self.course_mappings[course] = []
                    self.course_mappings[course].append((i, j))

    def _is_valid_index(self, row_idx, col_idx):
        """Check if the given indices are valid"""
        return (0 <= row_idx < self.num_rows and 
                1 <= col_idx < self.num_cols)  # Col 0 reserved for days

    def add_course_schedule(self, course_name, schedule):
        """
        Add a course with its complete schedule using indices
        schedule: List of tuples (row_idx, col_idx)
        Example: [(0, 1), (2, 2)] for slots in row 0, col 1 and row 2, col 2
        """
        # Validate indices
        for row_idx, col_idx in schedule:
            if not self._is_valid_index(row_idx, col_idx):
                raise ValueError(f"Invalid indices: ({row_idx}, {col_idx})")

        self.course_mappings[course_name] = schedule
        self._update_timetable()
        
    # Add this new method to the TimetableProcessor class
    def add_extra_row(self, row_name):
        """Add an extra row to the timetable"""
        # Create an empty row with the same columns as the existing DataFrame
        new_row = pd.DataFrame([[row_name] + [''] * (len(self.df.columns) - 1)], 
                            columns=self.df.columns)
        
        # Concatenate the new row with the existing DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Update the number of rows
        self.num_rows = len(self.df.index)
        
        # Update the timetable display
        self._update_timetable()

    def remove_course(self, course_name):
        """Remove a course and all its occurrences from the timetable"""
        if course_name in self.course_mappings:
            del self.course_mappings[course_name]
            self._update_timetable()

    def _update_timetable(self):
        """Update the timetable based on current course mappings"""
        # Clear existing timetable (preserve first column with days)
        for i in range(self.num_rows):
            for j in range(1, self.num_cols):
                self.df.iloc[i, j] = ''

        # Add all courses from mappings
        for course, indices in self.course_mappings.items():
            for row_idx, col_idx in indices:
                current_value = self.df.iloc[row_idx, col_idx]
                if current_value == '':
                    self.df.iloc[row_idx, col_idx] = course
                else:
                    self.df.iloc[row_idx, col_idx] = f"{current_value}/{course}"

    def get_course_schedule(self, course_name):
        """Get the schedule indices for a specific course"""
        return self.course_mappings.get(course_name, [])

    def get_time_slot_info(self, row_idx, col_idx):
        """Get the day and time slot for given indices"""
        if self._is_valid_index(row_idx, col_idx):
            day = self.df.iloc[row_idx, 0]
            time_slot = self.df.columns[col_idx]
            return {'day': day, 'time_slot': time_slot}
        return None

    def get_all_courses(self):
        """Get a list of all courses in the timetable"""
        return list(self.course_mappings.keys())

    def print_schedule_info(self, course_name):
        """Print human-readable schedule information for a course"""
        if course_name in self.course_mappings:
            print(f"Schedule for {course_name}:")
            for row_idx, col_idx in self.course_mappings[course_name]:
                info = self.get_time_slot_info(row_idx, col_idx)
                print(f"Day: {info['day']}, Time: {info['time_slot']}")
        else:
            print(f"Course {course_name} not found in timetable")


class SubjectTimetableManager:
    def __init__(self, page_content_div):
        self.page_content_div = page_content_div
        self.course_mapping = {}
        self.all_subjects = set()
        self.timings = None
        self.initialize_mappings()
        
    def initialize_mappings(self):
        """Create mappings for all courses and their subjects"""
        # Find all course divs
        course_divs = self.page_content_div.find_all('div', id=lambda x: x and x.startswith('prog_'))
        
        for div in course_divs:
            course_name = div.get('id').replace('prog_', '')
            processor = TimetableProcessor(course_name)
            
            try:
                # Extract and process timetable data
                timetable = processor.extract_div_content(self.page_content_div)
                extracted_data = processor.extract_timetable_data(timetable)
                processor.process_timetable(extracted_data)
                
                # Store timings from first course (assuming all courses have same timings)
                if self.timings is None:
                    self.timings = processor.df.columns[1:].tolist()  # Exclude 'Day' column
                
                # Create mapping for this course
                course_subjects = defaultdict(list)
                for i in range(processor.num_rows):
                    for j in range(1, processor.num_cols):  # Skip day column
                        subject = processor.df.iloc[i, j]
                        if pd.notna(subject) and subject.strip():
                            # Add to course mapping
                            timing = processor.df.columns[j]
                            day = processor.df.iloc[i, 0]
                            course_subjects[subject].append({
                                'day': day,
                                'timing': timing,
                                'row': i,
                                'col': j
                            })
                            # Add to all subjects set
                            self.all_subjects.add(subject)
                
                self.course_mapping[course_name] = dict(course_subjects)
                
            except Exception as e:
                print(f"Error processing course {course_name}: {str(e)}")
                
        # Convert subjects set to sorted list
        self.all_subjects = sorted(list(self.all_subjects))


def render_timetable(df, main_header_1, main_header_2, 
                     fig_size=(12, 12), font_size=14,
                     header1_fontsize=34, header2_fontsize=20):
    # Remove SAT row if it's empty
    df_clean = df.copy()
    if 'Day' in df_clean.columns:  # Check if DataFrame is reset (has Day as column)
        sat_row = df_clean[df_clean['Day'] == 'SAT']
        if sat_row.iloc[:, 1:].apply(lambda x: x == '').all().all():  # Check if all cells except 'Day' are empty
            df_clean = df_clean[df_clean['Day'] != 'SAT']
    else:  # If DataFrame is not reset (has Day as index)
        if 'SAT' in df_clean.index and df_clean.loc['SAT'].apply(lambda x: x == '').all():
            df_clean = df_clean.drop('SAT')

    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis('off')
    ax.text(0.5, 0.86, main_header_1, fontsize=header1_fontsize, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.8, main_header_2, fontsize=header2_fontsize, ha='center', va='center', fontweight='bold')
    table_bbox = [0.05, 0.15, 0.9, 0.6]

    cell_text = df.values.copy()
    for i in range(len(cell_text)):
        for j in range(len(cell_text[i])):
            if cell_text[i][j] and '/' in str(cell_text[i][j]):
                courses = cell_text[i][j].split('/')
                if len(courses) > 2:  # Check if more than two courses
                    cell_text[i][j] = '\n'.join(courses)
                else:
                    cell_text[i][j] = '/'.join(courses) # Changed to just \n

    mpl_table = ax.table(cellText=cell_text, bbox=table_bbox, colLabels=df_clean.columns, cellLoc='center')
    
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    col_width = table_bbox[2] / len(df_clean.columns)

    max_courses = 1
    for i in range(df_clean.shape[0]):
        for j in range(1, df_clean.shape[1]):
            content = str(df_clean.iloc[i, j])
            if content.strip():
                courses = content.split('\n')
                max_courses = max(max_courses, len(courses))

    total_height = table_bbox[3]
    row_height = total_height / (len(df_clean) + 1)
    base_row_height = row_height * (2 + (max_courses - 1) * 0.3)

    header_color = '#148cb1'

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
        cell.set_width(col_width)
        cell.set_height(base_row_height)

        cell.get_text().set_wrap(True)
        cell.get_text().set_verticalalignment('center')
        cell.set_text_props(weight='bold')

        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            if k[1] == 0:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(['#f1f1f2', 'w'][k[0] % 2])
        cell.get_text().set_wrap(True) 
        cell.get_text().set_verticalalignment('center')

    plt.tight_layout()
    return fig


def save_timetable(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="timetable.png">Download Timetable</a>'
    st.markdown(href, unsafe_allow_html=True)


def setup_subject_selector():
    try:
        url = "https://www.cmi.ac.in//practical/timetable.php"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        page_content_div = soup.find('div', class_='page_content')
        if page_content_div:
            return SubjectTimetableManager(page_content_div)
        else:
            st.error("Error: Could not find page content")
    except Exception as e:
        st.error(f"Error setting up selector: {str(e)}")


def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_subjects' not in st.session_state:
        st.session_state.selected_subjects = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'show_preview' not in st.session_state:
        st.session_state.show_preview = False
    if 'show_final' not in st.session_state:
        st.session_state.show_final = False
    if 'subject_selection_key' not in st.session_state:
        st.session_state.subject_selection_key = 0

def handle_subject_selection(new_selection):
    """Handle subject selection changes"""
    st.session_state.selected_subjects = new_selection
    st.session_state.current_df = None
    st.session_state.show_preview = False
    st.session_state.show_final = False

def create_editable_grid(df):
    """Create an editable grid using AgGrid"""
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True)
    gb.configure_column("Day", editable=False)
    gb.configure_grid_options(
        enableRangeSelection=True,
        suppressFieldDotNotation=True,
        suppressRowClickSelection=True
    )
    grid_options = gb.build()
    
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        height=400,
        key='grid'
    )
    
    return grid_response

def on_generate_preview():
    st.session_state.show_preview = True

def on_generate_final():
    st.session_state.show_final = True

def create_initial_dataframe(manager, selected_subjects):
    """Create initial DataFrame with selected subjects"""
    days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']
    df = pd.DataFrame(index=days, columns=manager.timings)
    df.index.name = 'Day'
    
    # Initialize all cells with empty lists to store multiple subjects
    for day in days:
        for timing in manager.timings:
            df.at[day, timing] = []
    
    # Add subjects to the appropriate cells
    for subject in selected_subjects:
        for course, subjects in manager.course_mapping.items():
            if subject in subjects:
                for slot in subjects[subject]:
                    current_subjects = df.at[slot['day'], slot['timing']]
                    if subject not in current_subjects:
                        current_subjects.append(subject)
                        df.at[slot['day'], slot['timing']] = current_subjects
    
    # Convert lists to strings with newline separators
    for day in days:
        for timing in manager.timings:
            subjects_list = df.at[day, timing]
            df.at[day, timing] = '/'.join(subjects_list) if subjects_list else ''
    
    # Remove SAT row if it's empty
    sat_row_empty = df.loc['SAT'].apply(lambda x: x == '').all()
    if sat_row_empty:
        df = df.drop('SAT')
    
    df.fillna('', inplace=True)
    df.reset_index(inplace=True)
    return df


def main():
    st.title("Timetable Selector")
    
    # Initialize session state
    initialize_session_state()
    
    manager = setup_subject_selector()

    if manager:
        sorted_subjects = sorted(list(manager.all_subjects))
        
        # Subject selector with callback
        selected = st.multiselect(
            "Select your subjects:",
            sorted_subjects,
            default=st.session_state.selected_subjects,
            key=f'subject_selector_{st.session_state.subject_selection_key}',
            on_change=handle_subject_selection,
            args=(st.session_state.selected_subjects,)
        )

        # Update selected subjects in session state using the callback
        if selected != st.session_state.selected_subjects:
            handle_subject_selection(selected)
            st.session_state.selected_subjects = selected
            st.session_state.current_df = None
            st.session_state.show_preview = False
            st.session_state.show_final = False

        # Generate Preview button
        col1, col2 = st.columns([2, 6])
        with col1:
            if st.button("Generate Preview", key='preview_button', use_container_width=True):
                on_generate_preview()

        # Show editable preview
        if st.session_state.show_preview:
            if st.session_state.current_df is None:
                st.session_state.current_df = create_initial_dataframe(manager, st.session_state.selected_subjects)
            
            st.subheader("Edit Timetable")
            grid_response = create_editable_grid(st.session_state.current_df)
            
            # Update current_df with edited data
            if grid_response['data'] is not None:
                st.session_state.current_df = pd.DataFrame(grid_response['data'])
            
            # Generate Final Timetable button
            if st.button("Generate Final Timetable", key='final_button'):
                on_generate_final()

        # Show final timetable and download button
        if st.session_state.show_final and st.session_state.current_df is not None:
            st.subheader("Final Timetable")
            fig = render_timetable(st.session_state.current_df, "Timetable", "")
            st.pyplot(fig)
            
            # Create download button
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            
            download_button = f'''
                <a href="data:image/png;base64,{b64}" 
                   download="timetable.png" 
                   class="button" 
                   style="display: inline-block; 
                          padding: 10px 20px; 
                          background-color: #148cb1; 
                          color: white; 
                          text-decoration: none; 
                          border-radius: 5px; 
                          margin-top: 10px;">
                    Download Timetable
                </a>
            '''
            st.markdown(download_button, unsafe_allow_html=True)
            plt.close(fig)

if __name__ == "__main__":
    main()