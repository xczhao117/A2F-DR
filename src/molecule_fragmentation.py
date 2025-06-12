import dill
from multiprocessing.pool import ThreadPool as Pool
from fragnet.dataset.fragments import get_3Dcoords2
from fragnet.dataset.data import CreateData


if __name__ == '__main__':
	maxiters = 500
	frag_type="brics"
	data_type = "exp1s"

	create_data = CreateData(
        data_type=data_type,
        create_bond_graph_data=True,
        add_dhangles=True,
    )

	atc2smile = dill.load(open('./data/raw/idx2SMILES.pkl', 'rb'))
	keys = atc2smile.keys() # 133 keys, the last 2 are not drugs and should be discarded.
	#print(list(keys))

	for idx, key in enumerate(keys):
		if idx > 130:
			break;
		#print(idx, key)
		smils_list = atc2smile[key]
		one_atc_data = []
		for smiles in smils_list:
			#print(smiles)
			res = get_3Dcoords2(smiles, maxiters=maxiters)
			if res != None:
				mol, conf_res = res
				for j in range(len(conf_res)):
					E = conf_res[j][1]
					conf = mol.GetConformer(id=j)
					x = [smiles, [E], mol, conf, frag_type]
					one_data = create_data.create_data_point(x)
					if one_data != None:
						one_atc_data.append(one_data)
		#print(one_atc_data)
		dill.dump(one_atc_data, open(f"./data/processed/{key}.pkl", 'wb'))