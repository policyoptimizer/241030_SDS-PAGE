# 특정 영역에 대해서만 진행하는 것으로 수정
# 다시 시작

import dash
from dash import html, dcc, Output, Input, State
import dash_canvas
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url
import plotly.express as px
import plotly.graph_objects as go
from skimage import io
import numpy as np
import pandas as pd
import base64
import cv2

# 이미지 업로드를 위한 함수
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img_array = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return img

# Dash 앱 생성 (주석 처리)
# app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("SDS-PAGE 밴드 IntDen 측정 웹앱"),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['이미지를 여기로 드래그하거나 클릭하여 업로드하세요.']),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    html.Button('분석 실행', id='analyze-button', n_clicks=0),
    html.Div(id='analysis-results')
])

@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents')
)
def update_output(contents):
    if contents is not None:
        img = parse_contents(contents)
        canvas_width = 800
        canvas_height = int(img.shape[0] * (canvas_width / img.shape[1]))
        return DashCanvas(
            id='canvas',
            width=canvas_width,
            height=canvas_height,
            image_content=contents,
            lineWidth=2,
            goButtonTitle='BB 추가',
            hide_buttons=['pencil', 'zoom', 'pan', 'line', 'redo', 'undo']
        )
    else:
        return html.Div(['이미지가 업로드되지 않았습니다.'])

@app.callback(
    Output('analysis-results', 'children'),
    Input('analyze-button', 'n_clicks'),
    State('canvas', 'json_data'),
    State('upload-image', 'contents')
)
def analyze(n_clicks, json_data, contents):
    if n_clicks > 0 and json_data and contents:
        img = parse_contents(contents)
        df_list = []
        bb_width = None
        bb_height = None
        for i, shape in enumerate(json_data['objects']):
            if shape['type'] == 'rect':
                if i == 0:
                    bb_width = int(shape['width'])
                    bb_height = int(shape['height'])
                left = int(shape['left'])
                top = int(shape['top'])
                width = bb_width
                height = bb_height
                roi = img[top:top+height, left:left+width]
                area = width * height
                mean = np.mean(roi)
                min_val = np.min(roi)
                max_val = np.max(roi)
                intden = np.sum(roi)
                rawintden = intden  # 동일하게 처리
                df_list.append({
                    '피크': f'Peak {i}',
                    'Area': area,
                    'Mean': mean,
                    'Min': min_val,
                    'Max': max_val,
                    'IntDen': intden,
                    'RawIntDen': rawintden
                })
        df = pd.DataFrame(df_list)
        # 마커 대비 % 계산
        marker_intden = df.loc[0, 'IntDen']
        df['Marker 대비 %'] = (df['IntDen'] / marker_intden) * 100
        # 결과 테이블
        table = dcc.Graph(
            figure=go.Figure(
                data=[go.Table(
                    header=dict(values=list(df.columns)),
                    cells=dict(values=[df[c] for c in df.columns])
                )]
            )
        )
        # 그래프
        graph = dcc.Graph(
            figure=px.bar(df, x='피크', y='IntDen', title='IntDen 값 그래프')
        )
        return [table, graph]
    else:
        return html.Div(['분석을 실행하려면 이미지를 업로드하고 BB를 지정한 후 "분석 실행" 버튼을 클릭하세요.'])

# 앱 실행 (주석 처리)
# if __name__ == '__main__':
#     app.run_server(debug=True)
