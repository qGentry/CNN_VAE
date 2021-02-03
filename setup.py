from setuptools import setup, find_packages


def main():
    with open("requirements.txt") as f:
        requirements = f.readlines()

    setup(
        name="CNN_VAE",
        version="0.1",
        author="Philipp Fisin",
        package_dir={"": "src"},
        packages=find_packages("src"),
        description="Variational autoencoder for generating image",
        install_requires=requirements,
    )


if __name__ == "__main__":
    main()
