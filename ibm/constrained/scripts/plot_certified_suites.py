"""Plot cumulative-best certified scores from constrained experiment suites."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def env(name, default=''):
    return os.environ.get(name, default).strip()


def finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def response_trace(data, record, response):
    columns = {name: index for index, name in enumerate(data['hist_columns'])}
    history = record['hist']
    seconds = np.asarray([finite_number(row[columns['seconds']]) for row in history])
    finite_times = np.flatnonzero(np.isfinite(seconds))
    if finite_times.size and env('INITIAL_AT_ZERO', '1') == '1':
        seconds[finite_times[0]] = 0.0
    score_name = f'score_y_{response}'
    if score_name in columns:
        score = np.asarray([finite_number(row[columns[score_name]]) for row in history])
    elif response == 'g':
        objective_name = next(name for name in ('phi_hat', 'f_y_g', 'conditional_phi') if name in columns)
        gap_name = next(name for name in ('gap_g', 'll_gap_y_g', 'll_cert_gap') if name in columns)
        objective = np.asarray([finite_number(row[columns[objective_name]]) for row in history])
        gap = np.asarray([finite_number(row[columns[gap_name]]) for row in history])
        score = objective + np.maximum(gap, 0.0)
    else:
        return None
    running = np.full_like(score, np.nan)
    best = np.inf
    for index, value in enumerate(score):
        if np.isfinite(value):
            best = min(best, value)
        if np.isfinite(best):
            running[index] = best
    return seconds, running


def interpolate_step(seconds, values, grid):
    result = np.full(grid.shape, np.nan)
    valid = np.isfinite(seconds) & np.isfinite(values)
    if not np.any(valid):
        return result
    seconds, values = seconds[valid], values[valid]
    order = np.argsort(seconds)
    seconds, values = seconds[order], values[order]
    indices = np.searchsorted(seconds, grid, side='right') - 1
    keep = indices >= 0
    result[keep] = values[indices[keep]]
    return result


def aggregate(data, response, grid, method_filter=None, center_stat='median'):
    curves = []
    for group in data['runs']:
        for record in group['runs']:
            if method_filter and record.get('method') != method_filter:
                continue
            trace = response_trace(data, record, response)
            if trace is not None:
                curves.append(interpolate_step(*trace, grid))
    if not curves:
        return None
    matrix = np.asarray(curves)
    with np.errstate(all='ignore'):
        center = (np.nanmean(matrix, axis=0) if center_stat == 'mean'
                  else np.nanmedian(matrix, axis=0))
        if center_stat == 'mean':
            spread = np.nanstd(matrix, axis=0)
            lower, upper = center - spread, center + spread
        else:
            lower = np.nanquantile(matrix, 0.25, axis=0)
            upper = np.nanquantile(matrix, 0.75, axis=0)
        return center, lower, upper, matrix


def main():
    result_paths = env('RESULTS').split()
    if not result_paths:
        raise ValueError('RESULTS must list at least one JSON file')
    labels = env('LABELS').split('|') if env('LABELS') else [Path(path).stem for path in result_paths]
    if len(labels) != len(result_paths):
        raise ValueError('LABELS must have one pipe-separated label per result')
    method_filters = env('METHOD_FILTERS').split('|') if env('METHOD_FILTERS') else [''] * len(result_paths)
    if len(method_filters) != len(result_paths):
        raise ValueError('METHOD_FILTERS must have one pipe-separated method per result')
    configured_colors = env('COLORS').split('|') if env('COLORS') else []
    if configured_colors and len(configured_colors) != len(result_paths):
        raise ValueError('COLORS must have one pipe-separated color per result')
    configured_styles = env('LINESTYLES').split('|') if env('LINESTYLES') else []
    if configured_styles and len(configured_styles) != len(result_paths):
        raise ValueError('LINESTYLES must have one pipe-separated style per result')
    responses = env('PLOT_RESPONSES', 'g').replace(',', ' ').split()
    center_stat = env('CENTER_STAT', 'median')
    if center_stat not in ('median', 'mean'):
        raise ValueError('CENTER_STAT must be median or mean')
    plot_stem = Path(env('PLOT_STEM', 'figures/certified_reporting/certified_suite'))
    plot_stem.parent.mkdir(parents=True, exist_ok=True)
    time_budget = max(float(json.load(open(path))['config']['time_budget']) for path in result_paths)
    grid = np.linspace(0.0, time_budget, int(time_budget) + 1)
    colors = configured_colors or plt.cm.viridis(np.linspace(0.05, 0.95, len(result_paths)))
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    summary_rows = []
    for series_index, (path, label, method_filter, color) in enumerate(
            zip(result_paths, labels, method_filters, colors)):
        data = json.load(open(path))
        for response in responses:
            aggregate_result = aggregate(data, response, grid, method_filter or None,
                                         center_stat=center_stat)
            if aggregate_result is None:
                continue
            median, lower, upper, matrix = aggregate_result
            linestyle = (configured_styles[series_index] if configured_styles
                         else '-' if response == 'g' else '--')
            response_label = label if len(responses) == 1 else f'{label} ($y_{response}$)'
            axis.plot(grid, median, color=color, linestyle=linestyle, linewidth=2.2, label=response_label)
            axis.fill_between(grid, lower, upper, color=color, alpha=0.10)
            final = matrix[:, -1]
            center_final = np.nanmean(final) if center_stat == 'mean' else np.nanmedian(final)
            if center_stat == 'mean':
                spread_final = np.nanstd(final)
                lower_final, upper_final = center_final - spread_final, center_final + spread_final
            else:
                lower_final = np.nanquantile(final, 0.25)
                upper_final = np.nanquantile(final, 0.75)
            summary_rows.append((Path(path).name, label, response, center_final,
                                 lower_final, upper_final,
                                 np.nanmin(final), np.sum(np.isfinite(final))))
    axis.set_xlabel('wall-clock time (s)', fontsize=float(env('LABEL_SIZE', '15')))
    axis.set_ylabel(env('Y_LABEL', r'$f(y)$ + LL gap'),
                    fontsize=float(env('LABEL_SIZE', '15')))
    axis.tick_params(axis='both', labelsize=float(env('TICK_SIZE', '12')))
    axis.set_yscale(env('Y_SCALE', 'log'))
    if env('X_MIN') and env('X_MAX'):
        axis.set_xlim(float(env('X_MIN')), float(env('X_MAX')))
    if env('Y_MIN') and env('Y_MAX'):
        axis.set_ylim(float(env('Y_MIN')), float(env('Y_MAX')))
    axis.grid(True, which='major', alpha=0.25)
    axis.legend(frameon=False, ncol=2, fontsize=float(env('LEGEND_SIZE', '9')),
                loc=env('LEGEND_LOC', 'best'))
    if env('NO_TITLE', '0') != '1':
        axis.set_title(env('PLOT_TITLE', 'Certified constrained objective'),
                       fontsize=float(env('TITLE_SIZE', '16')))
    fig.tight_layout()
    pdf_path = Path(str(plot_stem) + '.pdf')
    fig.savefig(pdf_path)
    spread_headers = (('mean_minus_std', 'mean_plus_std') if center_stat == 'mean'
                      else ('q25', 'q75'))
    header = (f'file\tlabel\tresponse\t{center_stat}\t{spread_headers[0]}\t'
              f'{spread_headers[1]}\tminimum\tfinite_runs\n')
    rows = [header] + ['\t'.join((file_name, label, response, *(f'{value:.8g}' for value in values[:-1]), str(int(values[-1])))) + '\n'
                       for file_name, label, response, *values in summary_rows]
    plot_stem.with_name(plot_stem.name + '_summary.tsv').write_text(''.join(rows))
    print('saved', pdf_path)
    print('saved', plot_stem.with_name(plot_stem.name + '_summary.tsv'))


if __name__ == '__main__':
    main()
