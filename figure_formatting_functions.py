def format_fraction_with_pi(x, pos):
    fract = Fraction(x).limit_denominator()
    if fract == 0:
        return "0"
    elif x == 1:
        return '$\\pi$'
    elif x == -1:
        return r'$\text{-}\pi$'
    else:
        if fract.numerator > 0:
            return f'$\\frac{{{fract.numerator}}}{{{fract.denominator}}}$' + '$\\pi$'
        else:
            return f'$-\\frac{{{abs(fract.numerator)}}}{{{fract.denominator}}}$' + '$\\pi$'

def format_fraction_with_pi_small(x, pos):
    fract = Fraction(x).limit_denominator()
    if fract == 0:
        return "0"
    elif x == 1:
        return '$\\pi$'
    elif x == -1:
        return '$-\\pi$'
    else:
        if fract.numerator > 0:
            if fract.numerator == 1:
                return f'$\\pi/{fract.denominator}$'
            else:
                return f'${fract.numerator}\\pi/{fract.denominator}$'
        else:
            return f'$-{abs(fract.numerator)}\\pi/{fract.denominator}$'

def format_fraction_with_r_jup(x, pos):
    fract = Fraction(x).limit_denominator()
    if fract == 0:
        return "0"
    elif x == 1:
        return '$R_{j}$'
    else:
        if fract.denominator == 1:
            return f'${fract.numerator}R_{{j}}$'
        else:
            return f'$\\frac{{{fract.numerator}}}{{{fract.denominator}}}$' + '$R_{j}$'
def format_fraction_with_r_jup_small(x, pos):
    fract = Fraction(x).limit_denominator()
    if fract == 0:
        return "0"
    elif x == 1:
        return '$R_{j}$'
    elif x == -1:
        return '$-R_{j}$'
    else:
        if fract.numerator == 1:
            return f'$R_{{j}}/{fract.denominator}$'
        elif fract.denominator == 1:
            return f'${fract.numerator}R_{{j}}$'
        elif fract.numerator > 0:
            return f'${fract.numerator}R_{{j}}/{fract.denominator}$'
        else:
            return f'$-{abs(fract.numerator)}R_{{j}}/{fract.denominator}$'

def generate_plot_style():
    plt.style.use('the_usual.mplstyle')
    fig, ax = plt.subplots(figsize=[6, 4])
    ax.xaxis.set_major_formatter(FuncFormatter(format_fraction_with_pi_small))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
    ax.set_xlabel(r'Phase angle $\alpha$')
    ax.set_ylabel(r'$L(\alpha)/L_{\odot}$')
    fig.tight_layout()
    return fig, ax