#include <stdio.h>
#include <dirent.h>
#include <string>
#include <iostream>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

int splitString(const std::string &s1) {
    for (int i = 0; i < s1.length(); i++) {
        if (std::isdigit(s1[i])) {
            return i;
        }
    }
}

bool numeric_string_compare(const std::string &s1, const std::string &s2) {
    int indexToSplitS1 = splitString(s1);
    int indexToSplitS2 = splitString(s2);
    string nameS1 = s1.substr(indexToSplitS1);
    string nameS2 = s2.substr(indexToSplitS2);
    int found = s1.find(".");
    if (nameS1.compare(nameS2)) {
        int numS1 = atoi(s1.substr(indexToSplitS1, s1.find(".") - indexToSplitS1).c_str());
        int numS2 = atoi(s2.substr(indexToSplitS2, s2.find(".") - indexToSplitS2).c_str());
        return numS1 < numS2;
    } else {
        std::string::const_iterator it1 = s1.begin(), it2 = s2.begin();
        return std::lexicographical_compare(it1, s1.end(), it2, s2.end());
    }
}

vector<std::string> getListImageNameFromPathFolder(string namePath) {
    struct dirent *de; // Pointer for directory entry

    // opendir() returns a pointer of DIR type.
    DIR *dr = opendir(namePath.c_str());

    if (dr == NULL) // opendir returns NULL if couldn't open directory
    {
        printf("Could not open current directory");
        return {};
    }

    // Refer http://pubs.opengroup.org/onlinepubs/7990989775/xsh/readdir.html
    std::vector<std::string> imagesName;
    while ((de = readdir(dr)) != NULL) {
        string name = de->d_name;
        if (name.find(".jpg") != string::npos) {
            imagesName.push_back(de->d_name);
        }
    }

    closedir(dr);

    sort(imagesName.begin(), imagesName.end(), numeric_string_compare);


    return imagesName;
}

int main(void) {
    string pathName = "../../../Desktop/src/data/Images/odo360nodoor/odo360nodoor_segmentation/";
    std::vector<std::string> imagesName = getListImageNameFromPathFolder(pathName);

    for (int i = 0; i < imagesName.size(); i++) {
        cout << imagesName[i] << endl;
    }
    return 0;
}
