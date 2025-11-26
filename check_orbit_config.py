import yaml

with open('configs/newtonian_tde.yaml') as f:
    c = yaml.safe_load(f)

print('Orbit parameters in config:')
orbit = c.get('orbit', {})
print(f'  pericentre: {orbit.get("pericentre", "NOT FOUND")}')
print(f'  eccentricity: {orbit.get("eccentricity", "NOT FOUND (will use default 0.95)")}')
print(f'  starting_distance: {orbit.get("starting_distance", "NOT FOUND (will use default 3.0)")}')

print('\nNOTE: Eccentricity and starting_distance use defaults if not in config.')
print('This is expected behavior - GUI will use 0.95 and 3.0 respectively.')
