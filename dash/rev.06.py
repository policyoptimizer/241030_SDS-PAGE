- 실행 버튼 삭제
실행 버튼을 누르지 않아도
사용자가 수정할 때마다 자동으로 변경 사항이 반영

- 임계값 조절 수준 변경
임계값 조절 슬라이드에 1부터 9단계까지 숫자를 표기했는데
실제 사용자가 마우스로 미세 조정이 가능하도록 함

- Crop 영역에 대해서 분석
Crop 된 영역에 대해서 분석을 해주세요

import dash
from dash import html, dcc, Input, Output, State
from dash_canvas import DashCanvas
import base64
import io
from PIL import Image
import numpy as np
import cv2
import json

#app = dash.Dash(__name__)

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
        dcc.Slider(
            id='threshold-slider',
            min=1,
            max=9,
            step=0.1,  # 미세 조정을 위해 step을 0.1로 설정
            value=5,
            marks={i: str(i) for i in range(1, 10)}  # 표시되는 마크는 1부터 9까지 정수
        ),
    ], style={'width': '80%', 'margin': '20px auto'}),
    html.Div([
        html.Label('크롭 영역 선택:'),
        DashCanvas(
            id='canvas',
            width=500,
            height=500,
            image_content=None,  # 초기에는 이미지 없음
            lineWidth=5,
            goButtonTitle='크롭 영역 선택',
            hide_buttons=['zoom', 'pan', 'reset'],
            tools=['rectangle'],  # 사각형 도구 활성화
        ),
    ], style={'width': '80%', 'margin': '20px auto'}),
    # '실행' 버튼을 제거하였습니다.
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
    threshold = int((threshold_value - 1) * (255 / 8))
    _, thresh = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
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
    Output('canvas', 'image_content'),
    Input('upload-image', 'contents')
)
def update_canvas(list_of_contents):
    if list_of_contents and len(list_of_contents) > 0:
        # 첫 번째 이미지를 캔버스에 표시
        image_contents = list_of_contents[0]
        return image_contents
    return None

@app.callback(
    Output('output-image-upload', 'children'),
    [Input('upload-image', 'contents'),
     Input('threshold-slider', 'value'),
     Input('canvas', 'json_data')],
    [State('upload-image', 'filename')]
)
def update_output(list_of_contents, threshold_value, json_data, list_of_names):
    if list_of_contents is not None:
        items = []
        crop_coords = None
        if json_data:
            # json_data에서 크롭 영역 좌표 추출
            shapes = json.loads(json_data)['objects']
            if shapes:
                shape = shapes[0]
                x = int(shape.get('left', 0))
                y = int(shape.get('top', 0))
                w = int(shape.get('width', 0))
                h = int(shape.get('height', 0))
                crop_coords = (x, y, w, h)
        for image_contents, filename in zip(list_of_contents, list_of_names):
            content_type, content_string = image_contents.split(',')
            decoded = base64.b64decode(content_string)
            image = Image.open(io.BytesIO(decoded))
            # 이미지 처리 및 IntDen 계산
            intden_values = calculate_intden(image, threshold_value, crop_coords)
            # 결과를 테이블로 표시
            items.append(
                html.Div([
                    html.H5(filename),
                    html.Img(src=image_contents, style={'width': '100%'}),
                    generate_table(intden_values)
                ], style={'width': '23%', 'display': 'inline-block', 'vertical-align': 'top', 'margin': '1%'})
            )
        # 아이템들을 부모 Div에 담아 flex 레이아웃 적용
        return html.Div(items, style={'display': 'flex', 'flex-wrap': 'wrap'})
    return ''

#if __name__ == '__main__':
#    app.run_server(debug=True)
