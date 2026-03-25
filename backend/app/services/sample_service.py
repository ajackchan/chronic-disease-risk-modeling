from __future__ import annotations


def get_demo_samples() -> list[dict]:
    return [
        {
            'id': 'sample-low',
            'label': '低风险样本',
            'inputs': {
                'ridageyr': 42,
                'bmxbmi': 22.3,
                'lbxglu': 92.0,
                'lbxtr': 95.0,
                'lbdhdd': 58.0,
                'lbdldl': 102.0,
                'lbxin': 6.5,
            },
        },
        {
            'id': 'sample-mid',
            'label': '中风险样本',
            'inputs': {
                'ridageyr': 56,
                'bmxbmi': 24.8,
                'lbxglu': 105.0,
                'lbxtr': 160.0,
                'lbdhdd': 45.0,
                'lbdldl': 118.0,
                'lbxin': 9.5,
            },
        },
        {
            'id': 'sample-high',
            'label': '高风险样本',
            'inputs': {
                'ridageyr': 64,
                'bmxbmi': 29.2,
                'lbxglu': 132.0,
                'lbxtr': 220.0,
                'lbdhdd': 36.0,
                'lbdldl': 145.0,
                'lbxin': 15.2,
            },
        },
    ]
