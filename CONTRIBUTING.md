# Contributing to PIVA

Thank you for considering contributing to **PIVA**! We value contributions of 
all kinds, from code improvements to documentation and bug fixes. This 
document outlines how you can contribute effectively and in alignment with the 
project's goals.

---

## How You Can Contribute

We welcome contributions in the following areas:
- **Interactive functionalities for data processing**: Help enhance or create 
user-friendly interactive tools.
- **Analysis methods**: Implement new algorithms for analysing ARPES spectra.
- **Maintenance**: Keep the codebase clean, efficient, and up-to-date.
- **Documentation**: Improve clarity, add examples, or expand explanations.
- **Testing**: Add or update tests to improve code reliability.
- **Bug fixes**: Identify and fix issues in the project.

---

## Getting Started

To set up the project locally:

1. **Create a Conda Virtual Environment**:
   - It’s recommended to use a Conda virtual environment for managing 
   dependencies:
     ```bash
     conda create -n piva-env python=3.10.8
     conda activate piva-env
     ```

2. **Fork and Clone**:
   - Fork the repository by clicking the **"Fork"** button at the top-right of 
   the page.
   - Clone your forked repository:
     ```bash
     git clone https://github.com/pudeIko/piva.git
     cd piva
     ```

3. **Install Dependencies**:
   - Install the required packages:
     ```bash
     pip install -e .[test]
     ```

4. **Run Tests**:
   - Verify that the existing codebase works as expected:
     ```bash
     pytest
     ```

---

## Branching Model

For any new implementations or changes, simply open a new branch. Name your 
branch descriptively based on the type of contribution, *e.g.*:

- `feature/new-analysis-method`
- `bugfix/fix-interactive-ui`

---

## Submitting Your Contribution

When submitting a pull request:

- Ensure your code adheres to PEP 8 coding standards.
- Write clear commit messages describing the changes.
- Ensure your contribution includes appropriate documentation in 
docstrings and, if necessary, more detailed explanations in the project's 
documentation files.
- If applicable, write new tests for your changes.

---

## Testing Guidelines

We use `pytest` for testing. To run the tests:

If you add new features, ensure they are covered by relevant test cases. 
Place tests in the `tests/` directory, following the existing structure.

---

## Issue Reporting

To report issues:

1. Check the Issues tab to see if the problem has already been reported.
2. If not, create a new issue with:
   - A clear title and description.
   - Steps to reproduce the issue.
   - Screenshots or code snippets, if applicable.

---

## Code of Conduct

This project is governed by a [Code of Conduct](CODE_OF_CONDUCT.md). By 
participating, you agree to uphold its principles of respect and collaboration.

---

## License

By contributing, you agree that your contributions will be licensed under this 
[license](LICENSE).

---

## Communication

For discussions, questions, or suggestions, feel free to 
[email](mailto:piva-project@proton.me) us.

