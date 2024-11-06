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
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        img = parse_contents(contents)
        canvas_width = 800
        canvas_height = int(img.shape[0] * (canvas_width / img.shape[1]))
        return html.Div([
            DashCanvas(
                id='canvas',
                width=canvas_width,
                height=canvas_height,
                image_content=contents,
                lineWidth=2,
                goButtonTitle='BB 추가',
                tool='rectangle',
                hide_buttons=['pencil', 'line', 'circle', 'redo', 'undo']
            ),
            html.Button('BB 삭제', id='delete-bb', n_clicks=0)
        ])
    else:
        return html.Div(['이미지가 업로드되지 않았습니다.'])

# BB 삭제 기능을 위한 콜백
@app.callback(
    Output('canvas', 'json_data'),
    Input('delete-bb', 'n_clicks'),
    State('canvas', 'json_data'),
    prevent_initial_call=True
)
def delete_bb(n_clicks, json_data):
    if n_clicks > 0 and json_data:
        json_data['objects'] = json_data['objects'][:-1]  # 마지막 BB 삭제
    return json_data

@app.callback(
    Output('analysis-results', 'children'),
    Input('analyze-button', 'n_clicks'),
    State('canvas', 'json_data'),
    State('upload-image', 'contents')
)
def analyze(n_clicks, json_data, contents):
    if n_clicks > 0:
        if json_data is None or 'objects' not in json_data or len(json_data['objects']) == 0:
            return html.Div(['BB가 지정되지 않았습니다.'])
        if contents is None:
            return html.Div(['이미지가 업로드되지 않았습니다.'])
        img = parse_contents(contents)
        df_list = []
        bb_width = None
        bb_height = None
        for i, shape in enumerate(json_data['objects']):
            if shape['type'] == 'rect':
                if bb_width is None and bb_height is None:
                    bb_width = int(shape['width'])
                    bb_height = int(shape['height'])
                else:
                    # 모든 BB의 크기를 첫 번째 BB와 동일하게 설정
                    shape['width'] = bb_width
                    shape['height'] = bb_height
                left = int(shape['left'])
                top = int(shape['top'])
                width = bb_width
                height = bb_height
                # 이미지 범위를 벗어나지 않도록 좌표 조정
                left = max(0, min(left, img.shape[1] - width))
                top = max(0, min(top, img.shape[0] - height))
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
        if len(df_list) == 0:
            return html.Div(['BB가 지정되지 않았습니다.'])
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
