from django import forms

FIELD_LABELS = {
    'C1': 'Age',
    'C2': 'BMI',
    'C3': 'Gestational age of delivery (weeks)',
    'C4': 'Gravidity',
    'C5': 'Parity',
    'C6': 'Initial onset symptoms (IOS)',
    'C7': 'Gestational age of IOS onset (weeks)',
    'C8': 'Interval from IOS onset to delivery (days)',
    'C9': 'Gestational age of hypertension onset (weeks)',
    'C10': 'Interval from hypertension onset to delivery (days)',
    'C11': 'Gestational age of edema onset (weeks)',
    'C12': 'Interval from edema onset to delivery (days)',
    'C13': 'Gestational age of proteinuria onset (weeks)',
    'C14': 'Interval from proteinuria onset to delivery (days)',
    'C15': 'Expectant treatment',
    'C16': 'Anti-hypertensive therapy before hospitalization',
    'C17': 'Past history',
    'C18': 'Maximum systolic blood pressure',
    'C19': 'Maximum diastolic blood pressure',
    'C20': 'Reasons for delivery',
    'C21': 'Mode of delivery',
    'C22': 'Maximum BNP value',
    'C23': 'Maximum values of creatinine',
    'C24': 'Maximum uric acid value',
    'C25': 'Maximum proteinuria value',
    'C26': 'Maximum total protein value',
    'C27': 'Maximum albumin value',
    'C28': 'Maximum ALT value',
    'C29': 'Maximum AST value',
    'C30': 'Maximum platelet value',
}

class IndividualPredictionForm(forms.Form):
    MODEL_CHOICES = [
        ('logistic', 'Regresión Logística'),
        ('ann', 'Red Neuronal'),
        ('svm', 'Máquina de Vectores de Soporte'),
        ('fcm', 'Mapa Cognitivo Difuso'),
    ]

    # Selector de modelo
    model = forms.ChoiceField(choices=MODEL_CHOICES, label="Modelo")

    # Campos C1 a C30
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(1, 31):
            field_name = f'C{i}'
            self.fields[field_name] = forms.FloatField(
                label=FIELD_LABELS.get(field_name, field_name),
                min_value=0,
                required=True
            )


class BatchPredictionForm(forms.Form):
    MODEL_CHOICES = [
        ('logistic', 'Regresión Logística'),
        ('ann', 'Red Neuronal'),
        ('svm', 'Máquina de Vectores de Soporte'),
        ('fcm', 'Mapa Cognitivo Difuso'),
    ]

    model = forms.ChoiceField(choices=MODEL_CHOICES, label="Modelo")
    file = forms.FileField(label="Archivo Excel (.xlsx)")
