import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import cv2
from dash.dependencies import Input, Output, State
import base64
import io
from PIL import Image

# app = dash.Dash(__name__)  # 주석 처리된 app 인스턴스

# 레이아웃 정의
app_layout = html.Div([
    html.H1("SDS-PAGE 밴드 Intensity 분석 웹앱"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            '이미지를 여기로 드래그하거나, ', html.A('파일 선택')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    dash_table.DataTable(id='intensity-table', style_table={'overflowX': 'auto'})
])

# 이미지 업로드 및 처리 함수
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    
    # 이미지를 OpenCV 형식으로 변환
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Grayscale 변환 및 임계처리
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    intensities = []
    for i, contour in enumerate(contours):
        # 각 윤곽선에 대해 마스크 생성
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # IntDen 계산 (픽셀의 합)
        int_den = cv2.sumElems(gray * (mask // 255))[0]
        intensities.append({'Peak': f'Peak {i+1}', 'IntDen': int_den})
    
    # DataFrame으로 변환
    df = pd.DataFrame(intensities)
    return df

# 콜백 정의
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('intensity-table', 'data'),
     Output('intensity-table', 'columns')],
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        # 업로드된 이미지 표시
        children = html.Div([
            html.H5(filename),
            html.Img(src=contents, style={'height': '300px'})
        ])
        
        # Intensity 분석
        df = parse_contents(contents)
        columns = [{'name': col, 'id': col} for col in df.columns]
        data = df.to_dict('records')
        
        return children, data, columns
    return None, [], []

# app.layout = app_layout  # 주석 처리된 레이아웃 설정

# if __name__ == '__main__':
#     app.run_server(debug=True)  # 주석 처리된 서버 실행 코드
