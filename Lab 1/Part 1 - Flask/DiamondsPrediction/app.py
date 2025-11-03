from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load label mappings
with open('label_mappings.pkl', 'rb') as f:
    label_mappings = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', 
                         cut_options=label_mappings['cut'],
                         color_options=label_mappings['color'],
                         clarity_options=label_mappings['clarity'])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            carat = float(request.form['carat'])
            cut = int(request.form['cut'])
            color = int(request.form['color'])
            clarity = int(request.form['clarity'])
            depth = float(request.form['depth'])
            table = float(request.form['table'])
            x = float(request.form['x'])
            y = float(request.form['y'])
            z = float(request.form['z'])
            
            # Create feature array
            features = np.array([[carat, cut, color, clarity, depth, table, x, y, z]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get the actual text values for display
            cut_text = [k for k, v in label_mappings['cut'].items() if v == cut][0]
            color_text = [k for k, v in label_mappings['color'].items() if v == color][0]
            clarity_text = [k for k, v in label_mappings['clarity'].items() if v == clarity][0]
            
            return render_template('index.html',
                                 prediction_text=f'Estimated Diamond Price: ${prediction:,.2f}',
                                 cut_options=label_mappings['cut'],
                                 color_options=label_mappings['color'],
                                 clarity_options=label_mappings['clarity'],
                                 input_values={
                                     'carat': carat,
                                     'cut': cut,
                                     'color': color,
                                     'clarity': clarity,
                                     'depth': depth,
                                     'table': table,
                                     'x': x,
                                     'y': y,
                                     'z': z
                                 },
                                 diamond_details={
                                     'cut': cut_text,
                                     'color': color_text,
                                     'clarity': clarity_text
                                 })
        except Exception as e:
            return render_template('index.html',
                                 prediction_text=f'Error: {str(e)}',
                                 cut_options=label_mappings['cut'],
                                 color_options=label_mappings['color'],
                                 clarity_options=label_mappings['clarity'])

if __name__ == '__main__':
    app.run(debug=True)