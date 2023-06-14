import matplotlib.pyplot as plt




data = {
    'h1': {
        'dual + attr + image': [80.1, 86.9, 87.7, 86.9, 85.9],
        'dual + attr': [76.8, 83.2, 84.1, 83.1, 82.0],
        'dual + image': [57.8, 66.3, 66.8, 69.3, 68.4],
        'dual': [55.0, 63.5, 65.8, 65.7, 64.7]
    },
    'mrr': {
        'dual + attr + image': [0.836, 0.895, 0.902, 0.894, 0.885],
        'dual + attr': [0.813, 0.866, 0.872, 0.860, 0.849],
        'dual + image': [0.643, 0.722, 0.726, 0.748, 0.739],
        'dual': [0.627, 0.698, 0.718, 0.713, 0.701]
    }
}


# subplots 1 * 2
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# fig.suptitle('YAGO3-10', fontsize=16)

# plot h1
ax = axes[0]
ax.set_title('h1')
ax.set_xlabel('epoch')
ax.set_ylabel('h1')
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['1', '2', '3', '4', '5'])
ax.set_ylim([50, 90])
ax.plot([1, 2, 3, 4, 5], data['h1']['dual + attr + image'], label='dual + attr + image', marker='o')
ax.plot([1, 2, 3, 4, 5], data['h1']['dual + attr'], label='dual + attr', marker='o')

ax.plot([1, 2, 3, 4, 5], data['h1']['dual + image'], label='dual + image', marker='o')
ax.plot([1, 2, 3, 4, 5], data['h1']['dual'], label='dual', marker='o')
ax.legend()

# plot mrr
ax = axes[1]
ax.set_title('mrr')
ax.set_xlabel('epoch')
ax.set_ylabel('mrr')
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['1', '2', '3', '4', '5'])
ax.set_ylim([0.5, 1])
ax.plot([1, 2, 3, 4, 5], data['mrr']['dual + attr + image'], label='dual + attr + image', marker='o')
ax.plot([1, 2, 3, 4, 5], data['mrr']['dual + attr'], label='dual + attr', marker='o')
ax.plot([1, 2, 3, 4, 5], data['mrr']['dual + image'], label='dual + image', marker='o')
ax.plot([1, 2, 3, 4, 5], data['mrr']['dual'], label='dual', marker='o')
ax.legend()

plt.savefig('yago3-10.png', dpi=300)
plt.show()