# 이걸로 AI LinkOn 1차 배포함

import dash
from dash import html, dcc, Input, Output, State
import base64
import io
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Dash 앱 인스턴스 주석 처리
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
            'width': '90%',  # 업로드 영역 가로 길이 유지
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px auto'
        },
        multiple=False  # 한 번에 하나의 이미지 업로드로 유지
    ),
    html.Div(id='output-image-upload'),
    # 슬라이더 컨테이너 항상 표시
    html.Div([
        html.Label('BB 너비 조정'),
        dcc.Slider(
            id='bb-width-slider',
            min=10,
            max=200,
            step=1,
            value=39,
            marks={10 + i * 21: str(i + 1) for i in range(10)}  # 1부터 10까지 표시
        ),
        html.Label('BB 높이 조정'),
        dcc.Slider(
            id='bb-height-slider',
            min=10,
            max=200,
            step=1,
            value=24,
            marks={10 + i * 21: str(i + 1) for i in range(10)}  # 1부터 10까지 표시
        ),
    ], id='slider-container', style={'width': '80%', 'margin': '20px auto'}),
    html.Button('실행', id='analyze-button', n_clicks=0),
    html.Div(id='analysis-results'),
])

def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    return image

def image_to_array(image):
    return np.array(image)

def image_array_to_base64_str(image_array):
    pil_img = Image.fromarray(image_array)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    encoded_image = base64.b64encode(buff.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded_image

@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        image = parse_image(contents)
        img_array = image_to_array(image)
        # 이미지 표시를 위해 base64 인코딩
        img_str = image_array_to_base64_str(img_array)
        # 이미지 표시
        fig = go.Figure()
        fig.add_trace(go.Image(z=img_array))
        fig.update_layout(
            dragmode='drawrect',
            newshape=dict(line_color='cyan', line_width=1),  # BB 굵기 조정
            title_text='이미지에서 관심 영역을 지정하세요.',
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
            width=1200,  # 그래프 가로 길이 유지
            height=img_array.shape[0] * (800 / img_array.shape[1])  # 이미지 높이 조절
        )
        return html.Div([
            html.H5(filename),
            dcc.Graph(
                id='graph',
                figure=fig,
                config={
                    'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
                    'edits': {
                        'shapePosition': True,  # BB 위치 이동 가능
                        'shapeResize': True,    # BB 크기 조정 가능
                        'shapeDelete': True     # BB 삭제 가능
                    }
                },
                style={'width': '90%', 'margin': '0 auto'}
            )
        ])
    return ''

@app.callback(
    Output('graph', 'figure'),
    Input('bb-width-slider', 'value'),
    Input('bb-height-slider', 'value'),
    State('graph', 'figure'),
    prevent_initial_call=True
)
def update_bb_size(width, height, fig):
    if 'shapes' in fig['layout']:
        shapes = fig['layout']['shapes']
        new_shapes = []
        for shape in shapes:
            x_center = (shape['x0'] + shape['x1']) / 2
            y_center = (shape['y0'] + shape['y1']) / 2
            new_x0 = x_center - width / 2
            new_x1 = x_center + width / 2
            new_y0 = y_center - height / 2
            new_y1 = y_center + height / 2
            shape['x0'] = new_x0
            shape['x1'] = new_x1
            shape['y0'] = new_y0
            shape['y1'] = new_y1
            new_shapes.append(shape)
        fig['layout']['shapes'] = new_shapes
    return fig

@app.callback(
    Output('analysis-results', 'children'),
    Input('analyze-button', 'n_clicks'),
    State('graph', 'figure'),
    State('upload-image', 'contents')
)
def analyze_image(n_clicks, fig, contents):
    if n_clicks > 0:
        if contents is None:
            return html.Div(['이미지를 업로드하세요.'])
        image = parse_image(contents)
        img_array = np.array(image.convert('L'))  # 그레이스케일 변환
        shapes = []
        if 'shapes' in fig['layout']:
            shapes = fig['layout']['shapes']
        else:
            return html.Div(['관심 영역을 지정하세요.'])
        if len(shapes) == 0:
            return html.Div(['관심 영역을 지정하세요.'])
        results = []
        for i, shape in enumerate(shapes):
            # shape 좌표 추출
            x0 = int(shape['x0'])
            y0 = int(shape['y0'])
            x1 = int(shape['x1'])
            y1 = int(shape['y1'])
            # 좌표 정렬
            xmin = int(min(x0, x1))
            xmax = int(max(x0, x1))
            ymin = int(min(y0, y1))
            ymax = int(max(y0, y1))
            # 이미지 범위 체크
            xmin = max(0, xmin)
            xmax = min(img_array.shape[1], xmax)
            ymin = max(0, ymin)
            ymax = min(img_array.shape[0], ymax)
            # ROI 추출
            roi = img_array[ymin:ymax, xmin:xmax]
            area = roi.size
            mean = np.mean(roi)
            min_val = np.min(roi)
            max_val = np.max(roi)
            intden = np.sum(roi)
            peak_name = 'Marker' if i == 0 else f'Peak {i}'
            results.append({
                '피크': peak_name,
                'Area': area,
                'Mean': round(mean, 1),
                'Min': round(min_val, 1),
                'Max': round(max_val, 1),
                'IntDen': intden,
                'Marker 대비 %': 0  # 일단 0으로 초기화
            })
        df = pd.DataFrame(results)
        # 마커 대비 % 계산
        marker_intden = df.loc[0, 'IntDen']
        df['Marker 대비 %'] = round((marker_intden / df['IntDen']) * 100, 1)
        # 결과 테이블
        table_fig = go.Figure(
            data=[go.Table(
                header=dict(values=list(df.columns)),
                cells=dict(values=[df[c] for c in df.columns])
            )],
            layout=go.Layout(title='분석 결과')
        )
        table = dcc.Graph(figure=table_fig)
        # 그래프
        graph_fig = px.bar(df, x='피크', y='IntDen', title='IntDen 값 그래프')
        graph = dcc.Graph(figure=graph_fig)
        return [table, graph]
    return ''

# Dash 앱 실행 코드 주석 처리
# if __name__ == '__main__':
#     app.run_server(debug=True)
