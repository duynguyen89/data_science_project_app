import streamlit as st
import pandas as pd
import pickle

# function cần thiết
def get_recommendations(df, id, cosine_sim, nums=5):
    # Get the index of the bike that matches the bike_id
    matching_indices = df.index[df['id'] == id].tolist()
    if not matching_indices:
        print(f"No house found with ID: {id}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all bikes with that bike
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the bikes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the nums most similar bikes (Ignoring the bike itself)
    sim_scores = sim_scores[1:nums+1]

    # Get the bike indices
    house_indices = [i[0] for i in sim_scores]

    # Return the top n most similar bikes as a DataFrame
    return df.iloc[house_indices]

# Hiển thị đề xuất ra bảng
def display_recommended_houses(recommended_houses, cols=5):
    for i in range(0, len(recommended_houses), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_houses):
                house = recommended_houses.iloc[i + j]
                with col:   
                    st.write(house['tieu_de'])                    
                    expander = st.expander(f"Description")
                    house_description = house['mo_ta']
                    truncated_description = ' '.join(house_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

# Đọc dữ liệu nhà ở từ file CSV
df_houses = pd.read_csv('house_samples.csv')
# Lấy 10 nhà ở ngẫu nhiên để hiển thị trong dropdown
random_houses = df_houses.head(n=10)
# print(random_houses)

st.session_state.random_houses = random_houses

# Open and read file to cosine_sim_new
with open('nha_cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

###### Giao diện Streamlit ######
### cho version cũ
st.image('nhatot.jpg', use_column_width=True)
### cho version mới
# st.image("nhatot.jpg", use_container_width=True)


# Kiểm tra xem 'selected_house_id' đã có trong session_state hay chưa
if 'selected_house_id' not in st.session_state:
    # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID nhà đầu tiên
    st.session_state.selected_house_id = None

# Theo cách cho người dùng chọn nhà từ dropdown
# Tạo một tuple cho mỗi nhà, trong đó phần tử đầu là tên và phần tử thứ hai là ID
house_options = [(row['tieu_de'], row['id']) for index, row in st.session_state.random_houses.iterrows()]
# st.session_state.random_houses
# Tạo một dropdown với options là các tuple này
selected_house = st.selectbox(
    "Chọn nhà bạn quan tâm:",
    options=house_options,
    format_func=lambda x: x[0]  # Hiển thị tên nhà
)
# Display the selected house
# st.write("Bạn đã chọn:", selected_house)

# Cập nhật session_state dựa trên lựa chọn hiện tại
st.session_state.selected_house_id = selected_house[1]

if st.session_state.selected_house_id:
    st.write("house_ID: ", st.session_state.selected_house_id)
    # Hiển thị thông tin nhà được chọn
    selected_house = df_houses[df_houses['id'] == st.session_state.selected_house_id]

    if not selected_house.empty:
        st.write('#### Bạn vừa chọn:')
        st.write('### ', selected_house['tieu_de'].values[0])

        house_description = selected_house['mo_ta'].values[0]
        truncated_description = ' '.join(house_description.split()[:100])
        st.write('##### Thông tin:')
        st.write(truncated_description, '...')

        st.write('##### Các nhà khác bạn cũng có thể quan tâm:')
        recommendations = get_recommendations(df_houses, st.session_state.selected_house_id, cosine_sim=cosine_sim_new, nums=3)
        display_recommended_houses(recommendations, cols=3)
    else:
        st.write(f"Không tìm thấy nhà với ID: {st.session_state.selected_house_id}")
