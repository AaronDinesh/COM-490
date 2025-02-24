#!/bin/bash

main() {
	local -r bin=$(dirname $(realpath ${BASH_SOURCE[0]}))
	cd -- "${bin}"
	if [[ ! -e ./data/promoter_sequences.csv || ! -e ./data/mystery_genes.csv ]]; then
		mkdir ./data/
		cd ./data
		wget -cO data-module-1a.zip https://drive.switch.ch/index.php/s/0BoBH3zyXpB5eEw/download
		unzip -qo data-module-1a.zip
	fi
}

main "$@"
