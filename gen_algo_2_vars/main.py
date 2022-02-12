from flask import Flask, redirect, request, url_for, render_template
from flask_bootstrap import Bootstrap
import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from gen_algo import execute_genetic_algo
#from refactor_new_algo import execute_genetic_algo
from video import create_video

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        both_selected = [True, True]
        try:
            check_maximum = request.form['find_max']
        except KeyError:
            both_selected[0] = False
        try:
            check_maximum = request.form['find_min']
        except KeyError:
            both_selected[1] = False

        if not both_selected[0] and not both_selected[1]:
            print('No se seleccionó nada')
            return redirect(url_for('.index'))
        if both_selected[0]:
            print('Se busca el máximo')
            find_max = True
        if both_selected[1]:
            print('Se busca el mínimo')
            find_max = False

        min_val_x = request.form['min_val_x']
        max_val_x = request.form['max_val_x']
        min_val_y = request.form['min_val_y']
        max_val_y = request.form['max_val_y']

        res_x = request.form['res_x']
        res_y = request.form['res_y']
        total_generations = request.form['total_generations']
        population_size = request.form['population_size']
        prob_mutate_individual = request.form['prob_mutate_individual']
        prob_mutate_gen = request.form['prob_mutate_gen']

        return redirect(url_for('.view_plots', min_val_x=min_val_x, min_val_y=min_val_y, max_val_x=max_val_x,max_val_y= max_val_y, res_x=res_x,res_y=res_y, total_generations=total_generations,find_max=find_max, population_size=population_size,prob_mut_i=prob_mutate_individual,prob_mut_gen=prob_mutate_gen))
    return render_template('index.html')

@app.route('/plots', methods=['GET'])
def view_plots():
    min_val_x = float(request.args['min_val_x'])
    min_val_y = float(request.args['min_val_y'])
    max_val_x = float(request.args['max_val_x'])
    max_val_y = float(request.args['max_val_y'])
    
    res_x = float(request.args['res_x'])
    res_y = float(request.args['res_y'])
    total_generations = int(request.args['total_generations'])
    population_size = int(request.args['population_size'])

    prob_mutate_individual = float(request.args['prob_mut_i'])
    prob_mutate_gen = float(request.args['prob_mut_gen'])
    find_max = request.args['find_max'] == 'True'

    result_algo = execute_genetic_algo(min_val_x, min_val_y, max_val_x, max_val_y, res_x, res_y, total_generations, find_max, population_size, prob_mutate_individual, prob_mutate_gen)
    
    if result_algo is not None:
        fig = Figure()
        fig.tight_layout()
        ax = fig.add_subplot(1,2,1)
        ax.plot(result_algo[0],result_algo[1], label=result_algo[4])
        ax.plot(result_algo[0],result_algo[2], label='Caso Promedio')
        ax.plot(result_algo[0],result_algo[3], label=result_algo[5])
        ax.legend(loc='best')
        ax.set_title(result_algo[7])

        ax_2 = fig.add_subplot(1,2,2)
        ax_2.plot(result_algo[6][1][0],result_algo[6][1][1],"go",color="red",)
        ax_2.set_xlabel("X")
        ax_2.set_ylabel("Y")
        ax_2.set_title("Mejor solución")
        
        
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        create_video()
        return render_template('my_plots.html', image=pngImageB64String, best_individual= result_algo[6], title_best = result_algo[8] )
    else:
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)