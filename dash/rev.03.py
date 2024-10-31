# 임계값 조정 슬라이드 추가
# crop 기능 추가

import dash
from dash import html, dcc, Input, Output, State
import base64
import io
from PIL import Image
import numpy as np
import cv2

# app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('SDS-PAGE 밴드 IntDen 추정'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            '이미지를 여기로 드래그하거나 ',
            html.A('파일 선택')
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px auto'
        },
        multiple=True  # 여러 이미지 업로드 허용
    ),
    html.Div([
        html.Label('임계값 조절:'),
        dcc.Slider(id='threshold-slider', min=0, max=255, step=1, value=128),
    ], style={'width': '80%', 'margin': '20px auto'}),
    html.Div([
        html.Label('크롭 영역 설정 (왼쪽 상단 x, y, 너비, 높이):'),
        dcc.Input(id='crop-x', type='number', placeholder='x', style={'margin-right': '10px'}),
        dcc.Input(id='crop-y', type='number', placeholder='y', style={'margin-right': '10px'}),
        dcc.Input(id='crop-w', type='number', placeholder='너비', style={'margin-right': '10px'}),
        dcc.Input(id='crop-h', type='number', placeholder='높이'),
    ], style={'width': '80%', 'margin': '20px auto'}),
    html.Div(id='output-image-upload'),
])

def calculate_intden(image, threshold_value, crop_coords):
    # 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # 이미지 크롭
    if crop_coords:
        x, y, w, h = crop_coords
        gray_image = gray_image[y:y+h, x:x+w]
    # 이진화 (사용자 임계값 적용)
    _, thresh = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    # 컨투어 찾기
    contours, _ = cv2.findContours(255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 각 컨투어에 대해 IntDen 계산 (면적 제외, 평균값만 사용)
    intden_values = []
    for i, cnt in enumerate(contours):
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
        mean_intensity = cv2.mean(gray_image, mask=mask)[0]
        intden_values.append({'밴드 번호': i+1, 'IntDen': mean_intensity})
    # 밴드 번호 순서대로 정렬 (x 좌표 기준)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours_with_info = zip(contours, intden_values, bounding_boxes)
    sorted_contours = sorted(contours_with_info, key=lambda x: x[2][0])  # x 좌표로 정렬
    sorted_intden_values = [item[1] for item in sorted_contours]
    return sorted_intden_values

def generate_table(data):
    table_header = [
        html.Thead(html.Tr([html.Th('밴드 번호'), html.Th('IntDen'), html.Th('마커대비 %')]))
    ]
    rows = []
    if data:
        marker_intden = data[0]['IntDen']  # 첫 번째 밴드를 마커로 간주
        for row in data:
            percent = (row['IntDen'] / marker_intden) * 100 if marker_intden != 0 else 0
            rows.append(html.Tr([
                html.Td(row['밴드 번호']),
                html.Td(f"{row['IntDen']:.2f}"),
                html.Td(f"{percent:.2f}%")
            ]))
    table_body = [html.Tbody(rows)]
    return html.Table(table_header + table_body, style={'width': '100%'})

@app.callback(
    Output('output-image-upload', 'children'),
    [Input('upload-image', 'contents'),
     Input('threshold-slider', 'value'),
     Input('crop-x', 'value'),
     Input('crop-y', 'value'),
     Input('crop-w', 'value'),
     Input('crop-h', 'value')],
    State('upload-image', 'filename')
)
def update_output(list_of_contents, threshold_value, crop_x, crop_y, crop_w, crop_h, list_of_names):
    if list_of_contents is not None:
        items = []
        crop_coords = None
        if None not in [crop_x, crop_y, crop_w, crop_h]:
            crop_coords = (int(crop_x), int(crop_y), int(crop_w), int(crop_h))
        for image_contents, filename in zip(list_of_contents, list_of_names):
            content_type, content_string = image_contents.split(',')
            decoded = base64.b64decode(content_string)
            image = Image.open(io.BytesIO(decoded))
            # 이미지 처리 및 IntDen 계산
            intden_values = calculate_intden(image, threshold_value, crop_coords)
            # 이미지 표시를 위해 base64 인코딩
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_data = f"data:image/png;base64,{img_str}"
            # 결과를 테이블로 표시
            items.append(
                html.Div([
                    html.H5(filename),
                    html.Img(src=img_data, style={'width': '100%'}),
                    generate_table(intden_values)
                ], style={'width': '23%', 'display': 'inline-block', 'vertical-align': 'top', 'margin': '1%'})
            )
        # 아이템들을 부모 Div에 담아 flex 레이아웃 적용
        return html.Div(items, style={'display': 'flex', 'flex-wrap': 'wrap'})
    return ''

# if __name__ == '__main__':
#     app.run_server(debug=True)
